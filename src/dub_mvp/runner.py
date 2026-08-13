from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from dub_mvp.localize import (
    LocalizationError,
    LocalizationPipeline,
    TranslationMetrics,
)
from dub_mvp.manifest import (
    Lease,
    MutationAborted,
    RunManifest,
    RunStatus,
    StageStatus,
    append_stage_event,
    begin_stage,
    complete_stage,
    fail_stage,
    mutate_manifest,
    retry_delay_seconds,
)
from dub_mvp.media import MediaIngestor, MediaToolError
from dub_mvp.observability import capture_resource_snapshot, resources_since
from dub_mvp.render import RenderError, RenderPipeline, RenderReport
from dub_mvp.synthesize import (
    SynthesisError,
    SynthesisMetrics,
    SynthesisPipeline,
)
from dub_mvp.storage import (
    DEFAULT_MINIMUM_FREE_BYTES,
    StorageCapacityError,
    require_stage_capacity,
)
from dub_mvp.transcribe import TranscriptionError, TranscriptionPipeline
from dub_mvp.utterances import UtteranceError, UtterancePipeline


HEAVY_STAGES = {"transcribe", "segment", "localize", "synthesize", "render"}
RUNNABLE_STAGES = {"ingest", *HEAVY_STAGES}

STAGE_RUN_STATUS = {
    "ingest": RunStatus.INGESTED,
    "transcribe": RunStatus.TRANSCRIBED,
    "segment": RunStatus.SEGMENTED,
    "localize": RunStatus.LOCALIZED,
    "synthesize": RunStatus.SYNTHESIZED,
    "render": RunStatus.RENDERED,
}


class JobRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageRequest:
    run_directory: Path
    stage: str
    glossary_path: Path | None = None
    translation_context_path: Path | None = None
    voice_reference_path: Path | None = None
    lease: Lease | None = None


@dataclass(frozen=True)
class StageInputs:
    source_path: str
    source_start_ms: int
    source_end_ms: int
    source_language: str
    target_language: str
    outputs: Mapping[str, str]
    attempt_count: int

    @property
    def duration_ms(self) -> int:
        return self.source_end_ms - self.source_start_ms


class JobRunner(Protocol):
    def submit_ingest(self, run_directory: Path) -> None:
        pass

    def submit_stage(self, request: StageRequest) -> None:
        pass


class LocalJobRunner:
    def __init__(
        self,
        *,
        ingestor: MediaIngestor | None = None,
        transcription_pipeline: Any | None = None,
        utterance_pipeline: Any | None = None,
        localization_pipeline: Any | None = None,
        synthesis_pipeline: Any | None = None,
        render_pipeline: Any | None = None,
        background: bool = True,
        minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
        disk_free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        self.ingestor = ingestor or MediaIngestor()
        self.transcription_pipeline = transcription_pipeline
        self.utterance_pipeline = utterance_pipeline
        self.localization_pipeline = localization_pipeline
        self.synthesis_pipeline = synthesis_pipeline
        self.render_pipeline = render_pipeline
        self.background = background
        self.minimum_free_bytes = minimum_free_bytes
        self.disk_free_bytes = disk_free_bytes

    def submit_ingest(self, run_directory: Path) -> None:
        self._submit(
            self._run_stage,
            StageRequest(run_directory=run_directory, stage="ingest"),
        )

    def submit_stage(self, request: StageRequest) -> None:
        if request.stage not in RUNNABLE_STAGES:
            raise JobRunnerError(f"Unknown stage: {request.stage}")
        self._submit(self._run_stage, request)

    def _submit(self, target: Any, argument: Any) -> None:
        if self.background:
            thread = threading.Thread(target=target, args=(argument,), daemon=True)
            thread.start()
        else:
            target(argument)

    def _run_stage(self, request: StageRequest) -> None:
        """Execute one stage without holding the manifest across the work.

        Every state transition is a short locked read-modify-write, so a
        heartbeat or cancellation landing mid-stage cannot invalidate the
        commit, and a failure is always recorded.
        """
        manifest = begin_stage(
            request.run_directory,
            request.stage,
            lease=request.lease,
        )
        if manifest is None:
            return

        # Reading the inputs happens inside the guard too: anything that can
        # raise between claiming a stage and finishing it must still record an
        # outcome, or the stage sits in RUNNING with no explanation.
        started = time.monotonic()
        resource_snapshot = capture_resource_snapshot()
        inputs: StageInputs | None = None
        try:
            inputs = _stage_inputs(manifest, request.stage)
            del manifest
            require_stage_capacity(
                request.run_directory,
                stage=request.stage,
                source_path=Path(inputs.source_path),
                minimum_free_bytes=self.minimum_free_bytes,
                free_bytes=self.disk_free_bytes,
            )
            outputs, models, media, stage_metadata = self._execute(
                request,
                inputs,
            )
        except (
            JobRunnerError,
            MediaToolError,
            TranscriptionError,
            UtteranceError,
            LocalizationError,
            SynthesisError,
            RenderError,
            StorageCapacityError,
        ) as error:
            fail_stage(
                request.run_directory,
                request.stage,
                error=str(error),
                lease=request.lease,
                error_class=type(error).__name__,
                retryable=(
                    request.lease is not None
                    and getattr(error, "retryable", True)
                ),
                retry_delay_seconds=retry_delay_seconds(
                    inputs.attempt_count if inputs else 1
                ),
                duration_seconds=time.monotonic() - started,
                resources=resources_since(resource_snapshot),
            )
            return
        except Exception as error:  # noqa: BLE001 - never leave a stage stuck
            # An unexpected error is a defect rather than a transient fault, so
            # it fails terminally. Recording it here keeps a stage from sitting
            # in RUNNING with no explanation, and keeps the worker loop alive
            # for every other run.
            fail_stage(
                request.run_directory,
                request.stage,
                error=f"{type(error).__name__}: {error}",
                lease=request.lease,
                error_class="unexpected_error",
                retryable=False,
                duration_seconds=time.monotonic() - started,
                resources=resources_since(resource_snapshot),
            )
            return

        complete_stage(
            request.run_directory,
            request.stage,
            lease=request.lease,
            outputs=outputs,
            run_status=STAGE_RUN_STATUS[request.stage],
            models=models,
            media=media,
            duration_seconds=time.monotonic() - started,
            provider=stage_metadata.get("provider"),
            input_fingerprint=stage_metadata.get("input_fingerprint"),
            cost_usd=stage_metadata.get("cost_usd"),
            record_cost="cost_usd" in stage_metadata,
            resources=resources_since(resource_snapshot),
        )

    def _execute(
        self,
        request: StageRequest,
        inputs: StageInputs,
    ) -> tuple[dict[str, str], dict[str, str], Any, dict[str, Any]]:
        if request.stage == "ingest":
            media, outputs = self.ingestor.ingest(
                source=Path(inputs.source_path),
                run_directory=request.run_directory,
                start_ms=inputs.source_start_ms,
                end_ms=inputs.source_end_ms,
            )
            return outputs, {}, media, {}
        if request.stage == "transcribe":
            transcript, _, outputs = (
                self.transcription_pipeline or TranscriptionPipeline()
            ).run(
                audio_path=Path(_required_output(inputs, "working_audio")),
                run_directory=request.run_directory,
                language=inputs.source_language,
                duration_ms=inputs.duration_ms,
            )
            return outputs, {"whisperx": transcript.model}, None, {}
        if request.stage == "segment":
            _, _, outputs = (
                self.utterance_pipeline or UtterancePipeline()
            ).run(
                transcript_path=Path(_required_output(inputs, "transcript")),
                segments_path=Path(_required_output(inputs, "segments")),
                run_directory=request.run_directory,
            )
            return outputs, {}, None, {}
        if request.stage == "localize":
            pipeline = self.localization_pipeline or LocalizationPipeline()
            _, outputs, model_name = pipeline.run(
                segments_path=Path(
                    _required_output(inputs, "translation_segments")
                ),
                run_directory=request.run_directory,
                source_language=inputs.source_language,
                target_language=inputs.target_language,
                glossary_path=request.glossary_path,
                context_path=request.translation_context_path,
            )
            metrics = TranslationMetrics.model_validate_json(
                Path(outputs["translation_metrics"]).read_text(encoding="utf-8")
            )
            return outputs, {"translator": model_name}, None, {
                "provider": metrics.provider,
                "input_fingerprint": metrics.configuration_fingerprint,
                "cost_usd": metrics.cost_usd,
            }
        if request.stage == "synthesize":
            if request.voice_reference_path is None:
                raise JobRunnerError("Voice reference is required.")
            _, outputs, model_name = (
                self.synthesis_pipeline or SynthesisPipeline()
            ).run(
                localized_segments_path=Path(
                    _required_output(inputs, "localized_segments")
                ),
                run_directory=request.run_directory,
                target_language=inputs.target_language,
                voice_reference_path=request.voice_reference_path,
            )
            stage_metadata: dict[str, Any] = {}
            metrics_path = outputs.get("synthesis_metrics")
            if metrics_path is not None:
                metrics = SynthesisMetrics.model_validate_json(
                    Path(metrics_path).read_text(encoding="utf-8")
                )
                stage_metadata = {
                    "provider": metrics.provider,
                    "input_fingerprint": metrics.configuration_fingerprint,
                }
            return outputs, {"tts": model_name}, None, stage_metadata
        _, outputs = (self.render_pipeline or RenderPipeline()).run(
            synthesized_segments_path=Path(
                _required_output(inputs, "synthesized_segments")
            ),
            source_segment_path=Path(_required_output(inputs, "source_segment")),
            run_directory=request.run_directory,
            duration_ms=inputs.duration_ms,
        )
        stage_metadata: dict[str, Any] = {"provider": "ffmpeg"}
        report_path = outputs.get("render_report")
        if report_path is not None:
            report = RenderReport.model_validate_json(
                Path(report_path).read_text(encoding="utf-8")
            )
            stage_metadata["input_fingerprint"] = (
                report.configuration_fingerprint
            )
        return outputs, {}, None, stage_metadata


class QueuedJobRunner:
    """Record work for an external worker without executing it in the web app."""

    def submit_ingest(self, run_directory: Path) -> None:
        self._queue(run_directory=run_directory, stage="ingest")

    def submit_stage(self, request: StageRequest) -> None:
        if request.stage not in HEAVY_STAGES:
            raise JobRunnerError(f"Unknown stage: {request.stage}")
        self._queue(
            run_directory=request.run_directory,
            stage=request.stage,
        )

    def _queue(
        self,
        *,
        run_directory: Path,
        stage: str,
    ) -> None:
        queued_run_id: list[str] = []

        def apply(manifest: RunManifest) -> None:
            record = manifest.stages[stage]
            if record.status in {StageStatus.QUEUED, StageStatus.RUNNING}:
                raise MutationAborted
            previous_status = record.status
            record.status = StageStatus.QUEUED
            record.started_at = None
            record.heartbeat_at = None
            record.lease_expires_at = None
            record.completed_at = None
            record.worker_id = None
            record.next_retry_at = None
            record.error_class = None
            record.error = None
            manifest.status = RunStatus.QUEUED
            append_stage_event(
                record,
                at=datetime.now(timezone.utc),
                event="queued",
                from_status=previous_status,
                to_status=StageStatus.QUEUED,
            )
            queued_run_id.append(manifest.run_id)

        mutate_manifest(run_directory, apply)
        if not queued_run_id:
            return
        _write_queue_event(
            run_directory=run_directory,
            run_id=queued_run_id[0],
            stage=stage,
        )


def _write_queue_event(
    *,
    run_directory: Path,
    run_id: str,
    stage: str,
) -> None:
    metadata = run_directory / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    event_path = metadata / "job-queue.jsonl"
    # This file is a compatibility/audit hint only. The manifest and its
    # structured event projection are authoritative; input paths are omitted
    # because they add sensitive local detail without helping queue recovery.
    payload = {
        "run_id": run_id,
        "stage": stage,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _stage_inputs(manifest: RunManifest, stage: str) -> StageInputs:
    record = manifest.stages.get(stage)
    return StageInputs(
        source_path=manifest.source_path,
        source_start_ms=manifest.source_start_ms,
        source_end_ms=manifest.source_end_ms,
        source_language=manifest.source_language,
        target_language=manifest.target_language,
        outputs=MappingProxyType(dict(manifest.outputs)),
        attempt_count=record.attempt_count if record else 1,
    )


def _required_output(inputs: StageInputs, name: str) -> str:
    path = inputs.outputs.get(name)
    if not path:
        raise JobRunnerError(f"Required output is missing: {name}")
    if not Path(path).is_file():
        raise JobRunnerError(f"Required output file is missing: {path}")
    return path

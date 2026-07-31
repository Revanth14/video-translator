from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from dub_mvp.localize import LocalizationError, LocalizationPipeline
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.media import MediaIngestor, MediaToolError
from dub_mvp.render import RenderError, RenderPipeline
from dub_mvp.synthesize import SynthesisError, SynthesisPipeline
from dub_mvp.transcribe import TranscriptionError, TranscriptionPipeline


HEAVY_STAGES = {"transcribe", "localize", "synthesize", "render"}


class JobRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageRequest:
    run_directory: Path
    stage: str
    glossary_path: Path | None = None
    voice_reference_path: Path | None = None


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
        localization_pipeline: Any | None = None,
        synthesis_pipeline: Any | None = None,
        render_pipeline: Any | None = None,
        background: bool = True,
    ) -> None:
        self.ingestor = ingestor or MediaIngestor()
        self.transcription_pipeline = transcription_pipeline
        self.localization_pipeline = localization_pipeline
        self.synthesis_pipeline = synthesis_pipeline
        self.render_pipeline = render_pipeline
        self.background = background

    def submit_ingest(self, run_directory: Path) -> None:
        self._submit(self._run_ingest, run_directory)

    def submit_stage(self, request: StageRequest) -> None:
        if request.stage not in HEAVY_STAGES:
            raise JobRunnerError(f"Unknown stage: {request.stage}")
        self._submit(self._run_stage, request)

    def _submit(self, target: Any, argument: Any) -> None:
        if self.background:
            thread = threading.Thread(target=target, args=(argument,), daemon=True)
            thread.start()
        else:
            target(argument)

    def _run_ingest(self, run_directory: Path) -> None:
        manifest = RunManifest.load(run_directory)
        stage = manifest.stages["ingest"]
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.now(timezone.utc)
        stage.completed_at = None
        stage.error = None
        manifest.status = RunStatus.RUNNING
        manifest.save(run_directory)
        try:
            metadata, outputs = self.ingestor.ingest(
                source=Path(manifest.source_path),
                run_directory=run_directory,
                start_ms=manifest.source_start_ms,
                end_ms=manifest.source_end_ms,
            )
        except MediaToolError as error:
            _fail_stage(manifest, run_directory, stage="ingest", error=error)
            return

        stage.status = StageStatus.COMPLETED
        stage.completed_at = datetime.now(timezone.utc)
        stage.outputs = outputs
        manifest.status = RunStatus.INGESTED
        manifest.media = metadata
        manifest.outputs.update(outputs)
        manifest.save(run_directory)

    def _run_stage(self, request: StageRequest) -> None:
        manifest = RunManifest.load(request.run_directory)
        record = manifest.stages[request.stage]
        record.status = StageStatus.RUNNING
        record.started_at = datetime.now(timezone.utc)
        record.completed_at = None
        record.error = None
        manifest.status = RunStatus.RUNNING
        manifest.save(request.run_directory)

        try:
            if request.stage == "transcribe":
                transcript, _, outputs = (
                    self.transcription_pipeline or TranscriptionPipeline()
                ).run(
                    audio_path=Path(_required_output(manifest, "working_audio")),
                    run_directory=request.run_directory,
                    language=manifest.source_language,
                    duration_ms=manifest.duration_ms,
                )
                manifest.models["whisperx"] = transcript.model
                manifest.status = RunStatus.TRANSCRIBED
            elif request.stage == "localize":
                _, outputs, model_name = (
                    self.localization_pipeline or LocalizationPipeline()
                ).run(
                    segments_path=Path(_required_output(manifest, "segments")),
                    run_directory=request.run_directory,
                    source_language=manifest.source_language,
                    target_language=manifest.target_language,
                    glossary_path=request.glossary_path,
                )
                manifest.models["translator"] = model_name
                manifest.status = RunStatus.LOCALIZED
            elif request.stage == "synthesize":
                if request.voice_reference_path is None:
                    raise JobRunnerError("Voice reference is required.")
                _, outputs, model_name = (
                    self.synthesis_pipeline or SynthesisPipeline()
                ).run(
                    localized_segments_path=Path(
                        _required_output(manifest, "localized_segments")
                    ),
                    run_directory=request.run_directory,
                    target_language=manifest.target_language,
                    voice_reference_path=request.voice_reference_path,
                )
                manifest.models["tts"] = model_name
                manifest.status = RunStatus.SYNTHESIZED
            else:
                _, outputs = (self.render_pipeline or RenderPipeline()).run(
                    synthesized_segments_path=Path(
                        _required_output(manifest, "synthesized_segments")
                    ),
                    source_segment_path=Path(
                        _required_output(manifest, "source_segment")
                    ),
                    run_directory=request.run_directory,
                    duration_ms=manifest.duration_ms,
                )
                manifest.status = RunStatus.RENDERED
        except (
            JobRunnerError,
            TranscriptionError,
            LocalizationError,
            SynthesisError,
            RenderError,
        ) as error:
            _fail_stage(
                manifest,
                request.run_directory,
                stage=request.stage,
                error=error,
            )
            return

        record.status = StageStatus.COMPLETED
        record.completed_at = datetime.now(timezone.utc)
        record.outputs = outputs
        manifest.outputs.update(outputs)
        manifest.save(request.run_directory)


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
            glossary_path=request.glossary_path,
            voice_reference_path=request.voice_reference_path,
        )

    def _queue(
        self,
        *,
        run_directory: Path,
        stage: str,
        glossary_path: Path | None = None,
        voice_reference_path: Path | None = None,
    ) -> None:
        manifest = RunManifest.load(run_directory)
        record = manifest.stages[stage]
        record.status = StageStatus.QUEUED
        record.started_at = None
        record.completed_at = None
        record.error = None
        manifest.status = RunStatus.QUEUED
        manifest.save(run_directory)
        _write_queue_event(
            run_directory=run_directory,
            run_id=manifest.run_id,
            stage=stage,
            glossary_path=glossary_path,
            voice_reference_path=voice_reference_path,
        )


def _write_queue_event(
    *,
    run_directory: Path,
    run_id: str,
    stage: str,
    glossary_path: Path | None,
    voice_reference_path: Path | None,
) -> None:
    metadata = run_directory / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    event_path = metadata / "job-queue.jsonl"
    payload = {
        "run_id": run_id,
        "stage": stage,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_directory),
        "glossary_path": str(glossary_path) if glossary_path else None,
        "voice_reference_path": (
            str(voice_reference_path) if voice_reference_path else None
        ),
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _required_output(manifest: RunManifest, name: str) -> str:
    path = manifest.outputs.get(name)
    if not path:
        raise JobRunnerError(f"Required output is missing: {name}")
    if not Path(path).is_file():
        raise JobRunnerError(f"Required output file is missing: {path}")
    return path


def _fail_stage(
    manifest: RunManifest,
    run_directory: Path,
    *,
    stage: str,
    error: Exception,
) -> None:
    message = str(error)
    record = manifest.stages[stage]
    record.status = StageStatus.FAILED
    record.error = message
    record.completed_at = datetime.now(timezone.utc)
    manifest.status = RunStatus.FAILED
    manifest.errors.append(message)
    manifest.save(run_directory)

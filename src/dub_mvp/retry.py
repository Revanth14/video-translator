from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from dub_mvp.artifacts import (
    ArtifactMetadata,
    ArtifactStatus,
    completed_artifact_metadata,
    fingerprint_inputs,
    relative_artifact_path,
    write_artifact_metadata,
)
from dub_mvp.manifest import (
    PIPELINE_STAGE_NAMES,
    MutationAborted,
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
    append_stage_event,
    mutate_manifest,
)


OPERATOR_RETRY_ATTEMPTS = 3


class RetryError(RuntimeError):
    retryable = False


class RetryStage(str, Enum):
    LOCALIZE = "localize"
    SYNTHESIZE = "synthesize"
    RENDER = "render"


class RetryReport(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    request_id: str
    run_id: str
    requested_at: datetime
    from_stage: RetryStage
    requested_utterance_ids: list[str]
    affected_utterance_ids: list[str]
    invalidated_sidecars: list[str]
    invalidated_artifact_kinds: list[str]
    queued_stage: str


_STAGE_OUTPUT_KEYS = {
    "localize": {
        "localization_raw",
        "localized_segments",
        "localized_segments_metadata",
        "translation_context",
        "translation_metrics",
    },
    "synthesize": {
        "synthesis_raw",
        "synthesis_raw_metadata",
        "synthesis_metrics",
        "synthesis_metrics_metadata",
        "duration_metrics",
        "duration_metrics_metadata",
        "duration_corrections",
        "duration_corrections_metadata",
        "synthesized_segments",
        "synthesized_segments_metadata",
        "speaker_voice_map",
        "speaker_voice_map_metadata",
    },
    "render": {
        "alignment_plan",
        "alignment_plan_metadata",
        "hindi_srt",
        "hindi_srt_metadata",
        "dubbed_audio",
        "dubbed_audio_metadata",
        "dubbed_video",
        "dubbed_video_metadata",
        "render_commands",
        "render_commands_metadata",
        "render_report",
        "render_report_metadata",
    },
}

_SYNTHESIS_AGGREGATE_KINDS = {
    "synthesized_segments",
    "synthesis_metrics",
    "synthesis_run",
    "duration_metrics",
    "duration_corrections",
}
_RENDER_KINDS = {
    "alignment_plan",
    "hindi_subtitles",
    "dubbed_audio",
    "dubbed_video",
    "render_commands",
    "render_report",
}
_BENCHMARK_KINDS = {
    "benchmark_json",
    "benchmark_markdown",
    "human_review_template",
}


def retry_run(
    run_directory: Path,
    *,
    from_stage: RetryStage,
    utterance_selectors: list[str] | None = None,
    now: datetime | None = None,
) -> RetryReport:
    """Invalidate requested work and queue the earliest affected stage.

    The command is deliberately two-phase. The first locked mutation fences
    workers by moving affected stages to ``INVALIDATED``. Sidecars are then
    invalidated without holding the manifest lock. A second locked mutation
    publishes the audit record and queues the requested stage. If the process
    dies between phases, no worker can consume partially invalidated work and
    rerunning this command safely completes the recovery.
    """
    run_directory = run_directory.resolve()
    moment = now or datetime.now(timezone.utc)
    manifest = RunManifest.load(run_directory)
    stage_name = from_stage.value
    stage_names = _affected_stage_names(stage_name)
    source_ids = _load_source_utterance_ids(manifest, from_stage)
    requested_ids = _resolve_utterance_ids(
        source_ids,
        utterance_selectors or [],
        allow_empty=from_stage == RetryStage.RENDER,
    )
    if from_stage == RetryStage.RENDER and utterance_selectors:
        raise RetryError(
            "Render is a run-level artifact; omit --utterances or retry from "
            "synthesize for selective work."
        )

    sidecars = _load_sidecars(run_directory)
    affected_ids = _expand_translation_batch_ownership(
        requested_ids,
        sidecars,
        from_stage=from_stage,
    )
    request_inputs = {
        "run_id": manifest.run_id,
        "manifest_revision": manifest.revision,
        "from_stage": stage_name,
        "requested_utterance_ids": requested_ids,
        "affected_utterance_ids": affected_ids,
    }
    request_id = fingerprint_inputs(request_inputs)[:20]

    _mark_retry_invalidated(
        run_directory,
        stage_names=stage_names,
        request_id=request_id,
        now=moment,
    )

    selected = _select_sidecars(
        run_directory,
        sidecars,
        from_stage=from_stage,
        affected_ids=affected_ids,
    )
    invalidated_paths: list[str] = []
    invalidated_kinds: set[str] = set()
    for metadata_path, metadata in selected:
        if metadata.status == ArtifactStatus.INVALID:
            continue
        metadata.status = ArtifactStatus.INVALID
        metadata.configuration = {
            **metadata.configuration,
            "operator_invalidation": {
                "request_id": request_id,
                "from_stage": stage_name,
                "utterance_ids": affected_ids,
            },
        }
        write_artifact_metadata(metadata_path, metadata)
        invalidated_paths.append(
            relative_artifact_path(metadata_path, run_directory)
        )
        invalidated_kinds.add(metadata.kind)

    report = RetryReport(
        request_id=request_id,
        run_id=manifest.run_id,
        requested_at=moment,
        from_stage=from_stage,
        requested_utterance_ids=requested_ids,
        affected_utterance_ids=affected_ids,
        invalidated_sidecars=sorted(invalidated_paths),
        invalidated_artifact_kinds=sorted(invalidated_kinds),
        queued_stage=stage_name,
    )
    report_path, report_metadata_path = _write_retry_report(
        report,
        run_directory=run_directory,
        request_inputs=request_inputs,
    )
    _queue_retry(
        run_directory,
        stage_names=stage_names,
        request_id=request_id,
        report_path=report_path,
        report_metadata_path=report_metadata_path,
        now=moment,
    )
    return report


def _affected_stage_names(from_stage: str) -> list[str]:
    start = PIPELINE_STAGE_NAMES.index(from_stage)
    return list(PIPELINE_STAGE_NAMES[start:])


def _load_source_utterance_ids(
    manifest: RunManifest,
    from_stage: RetryStage,
) -> list[str]:
    if from_stage == RetryStage.LOCALIZE:
        key = "translation_segments"
    elif from_stage == RetryStage.SYNTHESIZE:
        key = "localized_segments"
    else:
        key = "synthesized_segments"
    raw_path = manifest.outputs.get(key)
    if not raw_path:
        if from_stage == RetryStage.RENDER:
            return []
        raise RetryError(f"Cannot retry {from_stage.value}: missing output {key}.")
    path = Path(raw_path)
    if not path.is_file():
        raise RetryError(f"Cannot retry {from_stage.value}: missing file {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetryError(f"Cannot read utterance IDs from {path}: {error}") from error
    if isinstance(payload, dict):
        payload = payload.get("segments") or payload.get("utterances")
    if not isinstance(payload, list):
        raise RetryError(f"Expected an utterance list in {path}.")
    identifiers: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RetryError(f"Invalid utterance entry in {path}.")
        identifier = item.get("segment_id") or item.get("utterance_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise RetryError(f"Utterance entry is missing a stable ID in {path}.")
        identifiers.append(identifier.strip())
    if len(identifiers) != len(set(identifiers)):
        raise RetryError(f"Duplicate stable utterance IDs in {path}.")
    return identifiers


def _resolve_utterance_ids(
    identifiers: list[str],
    selectors: list[str],
    *,
    allow_empty: bool,
) -> list[str]:
    if not selectors:
        if identifiers:
            return identifiers
        if allow_empty:
            return []
        raise RetryError("No utterances are available to retry.")
    resolved: list[str] = []
    for raw in selectors:
        selector = raw.strip()
        if not selector:
            continue
        if selector in identifiers:
            match = selector
        elif selector.isdigit():
            requested_number = int(selector)
            matches = [
                identifier
                for identifier in identifiers
                if (
                    (suffix := re.search(r"(\d+)$", identifier)) is not None
                    and int(suffix.group(1)) == requested_number
                )
            ]
            if len(matches) != 1:
                if not matches:
                    raise RetryError(f"Unknown utterance selector: {selector}")
                raise RetryError(
                    f"Ambiguous utterance selector {selector}: {matches}"
                )
            match = matches[0]
        else:
            raise RetryError(f"Unknown utterance selector: {selector}")
        if match not in resolved:
            resolved.append(match)
    if not resolved and not allow_empty:
        raise RetryError("At least one utterance selector is required.")
    return resolved


def _load_sidecars(
    run_directory: Path,
) -> list[tuple[Path, ArtifactMetadata]]:
    loaded: list[tuple[Path, ArtifactMetadata]] = []
    for path in sorted(run_directory.glob("**/*.meta.json")):
        try:
            metadata = ArtifactMetadata.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            # Missing/corrupt proof is already non-reusable. It does not need
            # to make an operator retry fail before the stage can regenerate.
            continue
        loaded.append((path, metadata))
    return loaded


def _expand_translation_batch_ownership(
    requested_ids: list[str],
    sidecars: list[tuple[Path, ArtifactMetadata]],
    *,
    from_stage: RetryStage,
) -> list[str]:
    if from_stage != RetryStage.LOCALIZE:
        return requested_ids
    affected = list(requested_ids)
    requested = set(requested_ids)
    for _, metadata in sidecars:
        if metadata.kind != "translation_batch":
            continue
        owned = metadata.configuration.get("owned_ids", [])
        if isinstance(owned, list) and requested.intersection(owned):
            for identifier in owned:
                if isinstance(identifier, str) and identifier not in affected:
                    affected.append(identifier)
    return affected


def _select_sidecars(
    run_directory: Path,
    sidecars: list[tuple[Path, ArtifactMetadata]],
    *,
    from_stage: RetryStage,
    affected_ids: list[str],
) -> list[tuple[Path, ArtifactMetadata]]:
    selected: list[tuple[Path, ArtifactMetadata]] = []
    utterance_directories = {
        f"u-{fingerprint_inputs({'utterance_id': identifier})[:16]}"
        for identifier in affected_ids
    }
    affected = set(affected_ids)
    for path, metadata in sidecars:
        choose = False
        if from_stage == RetryStage.LOCALIZE:
            owned = metadata.configuration.get("owned_ids", [])
            choose = metadata.kind == "translation_batch" and (
                isinstance(owned, list) and bool(affected.intersection(owned))
            )
            choose = choose or metadata.kind == "localized_segments"
        if from_stage in {RetryStage.LOCALIZE, RetryStage.SYNTHESIZE}:
            relative_parts = path.relative_to(run_directory).parts
            choose = choose or (
                bool(utterance_directories.intersection(relative_parts))
                and metadata.kind
                in {
                    "speech_audio",
                    "speech_result",
                    "duration_corrected_audio",
                    "duration_fit_result",
                }
            )
            choose = choose or metadata.kind in _SYNTHESIS_AGGREGATE_KINDS
        if from_stage in {
            RetryStage.LOCALIZE,
            RetryStage.SYNTHESIZE,
            RetryStage.RENDER,
        }:
            choose = choose or metadata.kind in _RENDER_KINDS
            choose = choose or metadata.kind in _BENCHMARK_KINDS
        if choose:
            selected.append((path, metadata))
    return selected


def _mark_retry_invalidated(
    run_directory: Path,
    *,
    stage_names: list[str],
    request_id: str,
    now: datetime,
) -> None:
    accepted: list[bool] = []

    def apply(manifest: RunManifest) -> None:
        running = [
            name
            for name in stage_names
            if manifest.stages[name].status == StageStatus.RUNNING
        ]
        if running:
            raise RetryError(
                "Cannot invalidate active work; wait for or cancel stages: "
                + ", ".join(running)
            )
        for name in stage_names:
            record = manifest.stages[name]
            previous = record.status
            record.status = StageStatus.INVALIDATED
            record.next_retry_at = None
            record.lease_expires_at = None
            record.worker_id = None
            append_stage_event(
                record,
                at=now,
                event="operator_retry_invalidated",
                from_status=previous,
                to_status=StageStatus.INVALIDATED,
                detail=f"retry_request={request_id}",
            )
        manifest.status = RunStatus.QUEUED
        accepted.append(True)

    mutate_manifest(run_directory, apply)
    if not accepted:
        raise MutationAborted


def _queue_retry(
    run_directory: Path,
    *,
    stage_names: list[str],
    request_id: str,
    report_path: Path,
    report_metadata_path: Path,
    now: datetime,
) -> None:
    start_stage = stage_names[0]

    def apply(manifest: RunManifest) -> None:
        for index, name in enumerate(stage_names):
            record = manifest.stages[name]
            previous = record.status
            _remove_stage_outputs(manifest, name, record)
            record.status = (
                StageStatus.QUEUED if index == 0 else StageStatus.PENDING
            )
            record.retryable = True
            record.max_attempts = max(
                record.max_attempts,
                record.attempt_count + OPERATOR_RETRY_ATTEMPTS,
            )
            record.next_retry_at = None
            record.started_at = None
            record.heartbeat_at = None
            record.lease_expires_at = None
            record.completed_at = None
            record.worker_id = None
            record.error_class = None
            record.error = None
            record.duration_seconds = None
            record.resources = None
            record.input_fingerprint = None
            record.cost_usd = None
            append_stage_event(
                record,
                at=now,
                event=(
                    "operator_retry_queued"
                    if index == 0
                    else "operator_retry_waiting"
                ),
                from_status=previous,
                to_status=record.status,
                detail=f"retry_request={request_id}",
            )
        for key in list(manifest.outputs):
            if key.startswith("benchmark_") or key.startswith("human_review_"):
                manifest.outputs.pop(key, None)
        manifest.outputs["latest_retry"] = str(report_path.resolve())
        manifest.outputs["latest_retry_metadata"] = str(
            report_metadata_path.resolve()
        )
        manifest.status = RunStatus.QUEUED

    mutate_manifest(run_directory, apply)


def _remove_stage_outputs(
    manifest: RunManifest,
    stage_name: str,
    record: StageRecord,
) -> None:
    keys = set(record.outputs) | _STAGE_OUTPUT_KEYS.get(stage_name, set())
    for key in keys:
        manifest.outputs.pop(key, None)
    record.outputs = {}


def _write_retry_report(
    report: RetryReport,
    *,
    run_directory: Path,
    request_inputs: dict[str, Any],
) -> tuple[Path, Path]:
    directory = run_directory / "metadata" / "retries"
    path = directory / f"retry-{report.request_id}.json"
    metadata_path = path.with_name(path.name + ".meta.json")
    _write_json(path, report.model_dump(mode="json"))
    write_artifact_metadata(
        metadata_path,
        completed_artifact_metadata(
            artifact_id=f"operator_retry_{report.request_id}",
            kind="operator_retry",
            path=path,
            root=run_directory,
            inputs=request_inputs,
            provider="operator",
            configuration={
                "from_stage": report.from_stage.value,
                "utterance_ids": report.affected_utterance_ids,
            },
        ),
    )
    return path, metadata_path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

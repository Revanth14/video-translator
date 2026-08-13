from __future__ import annotations

import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dub_mvp.manifest import (
    PIPELINE_STAGE_NAMES,
    ResourceUsage,
    RunError,
    RunManifest,
    StageStatus,
    manifest_lock,
    redact_sensitive_text,
)


EVENT_LOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResourceSnapshot:
    monotonic_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    max_rss_mb: float


class WorkItemObservation(BaseModel):
    kind: str
    work_item_id: str
    status: str
    attempt_count: int = Field(ge=0)
    failed_attempts: int = Field(ge=0)
    provider: str | None = None
    model: str | None = None
    latency_seconds: float = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    history_files: list[str] = Field(default_factory=list)
    read_errors: list[str] = Field(default_factory=list)


class RunStatusDocument(BaseModel):
    schema_version: int = 1
    manifest_schema_version: int
    manifest_revision: int
    run_id: str
    status: str
    source: str
    range_ms: list[int]
    source_language: str
    target_language: str
    created_at: str
    updated_at: str
    stages: dict[str, str]
    stage_details: dict[str, dict[str, Any]]
    progress: dict[str, Any]
    metrics: dict[str, Any]
    work_items: dict[str, list[WorkItemObservation]]
    configuration: dict[str, Any]
    timings_seconds: dict[str, float]
    resources: dict[str, Any]
    cost: dict[str, Any]
    outputs: dict[str, str]
    errors: list[RunError]
    recent_events: list[dict[str, Any]]
    event_log: str


def capture_resource_snapshot() -> ResourceSnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = float(usage.ru_maxrss)
    # macOS reports bytes; Linux and the common BSDs report KiB.
    max_rss_mb = (
        max_rss / (1024 * 1024)
        if sys.platform == "darwin"
        else max_rss / 1024
    )
    return ResourceSnapshot(
        monotonic_seconds=time.monotonic(),
        cpu_user_seconds=usage.ru_utime,
        cpu_system_seconds=usage.ru_stime,
        max_rss_mb=max_rss_mb,
    )


def resources_since(snapshot: ResourceSnapshot) -> ResourceUsage:
    current = capture_resource_snapshot()
    return ResourceUsage(
        wall_seconds=max(
            0.0, current.monotonic_seconds - snapshot.monotonic_seconds
        ),
        cpu_user_seconds=max(
            0.0, current.cpu_user_seconds - snapshot.cpu_user_seconds
        ),
        cpu_system_seconds=max(
            0.0, current.cpu_system_seconds - snapshot.cpu_system_seconds
        ),
        max_rss_mb=current.max_rss_mb,
    )


def build_run_status(run_directory: Path) -> RunStatusDocument:
    run_directory = run_directory.resolve()
    manifest, events = _load_manifest_and_events(run_directory)
    work_items = {
        "translation_batches": _load_work_items(
            run_directory,
            pattern="translation/batches/*.attempts.json",
            kind="translation_batch",
            identifier_key="batch_id",
        ),
        "speech_utterances": _load_work_items(
            run_directory,
            pattern="speech/utterances/**/*.attempts.json",
            kind="speech_utterance",
            identifier_key="utterance_id",
        ),
        "duration_fits": _load_work_items(
            run_directory,
            pattern="speech/duration/**/*.attempts.json",
            kind="duration_fit",
            identifier_key="utterance_id",
        ),
    }
    stage_details = {
        name: _stage_detail(record)
        for name, record in manifest.stages.items()
    }
    stage_costs = {
        name: record.cost_usd
        for name, record in manifest.stages.items()
        if record.cost_usd is not None
    }
    resource_stages = {
        name: record.resources.model_dump(mode="json")
        for name, record in manifest.stages.items()
        if record.resources is not None
    }
    total_stages = len(PIPELINE_STAGE_NAMES)
    completed_stages = sum(
        manifest.stages[name].status == StageStatus.COMPLETED
        for name in PIPELINE_STAGE_NAMES
    )
    translation_segments = _json_list(
        manifest.outputs.get("translation_segments")
        or manifest.outputs.get("segments")
    )
    localized_segments = _json_list(
        manifest.outputs.get("localized_segments")
    )
    synthesized_segments = _json_list(
        manifest.outputs.get("synthesized_segments")
    )
    alignment = _json_value(manifest.outputs.get("alignment_plan"))
    duration_metrics = _json_value(manifest.outputs.get("duration_metrics"))
    render_report = _json_value(manifest.outputs.get("render_report"))
    benchmark_report = _json_value(manifest.outputs.get("benchmark_json"))
    needs_review = 0
    if isinstance(alignment, dict):
        needs_review = sum(
            isinstance(item, dict) and bool(item.get("needs_review"))
            for item in alignment.get("segments", [])
        )
    if isinstance(duration_metrics, dict):
        needs_review = max(
            needs_review,
            int(duration_metrics.get("human_review_required_count", 0)),
        )
    current_stage = next(
        (
            name
            for name in PIPELINE_STAGE_NAMES
            if manifest.stages[name].status
            in {StageStatus.RUNNING, StageStatus.QUEUED, StageStatus.FAILED}
        ),
        None,
    )
    stage_attempts = sum(
        len(record.attempts) for record in manifest.stages.values()
    )
    work_attempts = sum(
        item.attempt_count
        for group in work_items.values()
        for item in group
    )
    errors = manifest.error_records or _derived_errors(manifest)
    return RunStatusDocument(
        manifest_schema_version=manifest.schema_version,
        manifest_revision=manifest.revision,
        run_id=manifest.run_id,
        status=manifest.status.value,
        source=manifest.source_path,
        range_ms=[manifest.source_start_ms, manifest.source_end_ms],
        source_language=manifest.source_language,
        target_language=manifest.target_language,
        created_at=manifest.created_at.isoformat(),
        updated_at=manifest.updated_at.isoformat(),
        stages={
            name: record.status.value
            for name, record in manifest.stages.items()
        },
        stage_details=stage_details,
        progress={
            "stages": {
                "completed": completed_stages,
                "total": total_stages,
                "percent": round(
                    completed_stages / total_stages * 100, 1
                ),
                "current": current_stage,
            },
            "utterances": {
                "total": len(translation_segments),
                "localized": len(localized_segments),
                "synthesized": len(synthesized_segments),
            },
            "attempts": {
                "stage": stage_attempts,
                "work_item": work_attempts,
                "total": stage_attempts + work_attempts,
            },
        },
        metrics={
            "segments": len(translation_segments),
            "localized": len(localized_segments),
            "synthesized": len(synthesized_segments),
            "needs_review": needs_review,
            "duration_timing": duration_metrics,
            "render_validation": (
                render_report.get("validation")
                if isinstance(render_report, dict)
                else None
            ),
            "benchmark_release_gate": (
                benchmark_report.get("release_gate_status")
                if isinstance(benchmark_report, dict)
                else None
            ),
            "duration": _duration_label(manifest.duration_ms),
            "has_video": _output_inside_run(
                run_directory, manifest.outputs.get("dubbed_video")
            ),
        },
        work_items=work_items,
        configuration={
            "source_language": manifest.source_language,
            "target_language": manifest.target_language,
            "range_ms": [manifest.source_start_ms, manifest.source_end_ms],
            "models": manifest.models,
            "stages": {
                name: {
                    "provider": record.provider,
                    "model": record.model,
                    "input_fingerprint": record.input_fingerprint,
                    "max_attempts": record.max_attempts,
                }
                for name, record in manifest.stages.items()
            },
        },
        timings_seconds=manifest.timings_seconds,
        resources={
            "stages": resource_stages,
            "total_wall_seconds": sum(
                item["wall_seconds"] for item in resource_stages.values()
            ),
            "total_cpu_user_seconds": sum(
                item["cpu_user_seconds"]
                for item in resource_stages.values()
            ),
            "total_cpu_system_seconds": sum(
                item["cpu_system_seconds"]
                for item in resource_stages.values()
            ),
            "peak_rss_mb": max(
                (
                    item["max_rss_mb"]
                    for item in resource_stages.values()
                ),
                default=0.0,
            ),
        },
        cost={
            "reported_usd": sum(stage_costs.values()),
            "by_stage_usd": stage_costs,
            "stages_reported": len(stage_costs),
            "stages_unknown": [
                name
                for name, record in manifest.stages.items()
                if record.status == StageStatus.COMPLETED
                and record.cost_usd is None
            ],
        },
        outputs=manifest.outputs,
        errors=errors,
        recent_events=events[-25:],
        event_log=str(event_log_path(run_directory).relative_to(run_directory)),
    )


def write_run_event_log(manifest: RunManifest, run_directory: Path) -> Path:
    path = event_log_path(run_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    records = _event_records(manifest)
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    return path


def load_run_events(run_directory: Path) -> list[dict[str, Any]]:
    """Read the JSONL projection, rebuilding corrupt or stale projections."""
    run_directory = run_directory.resolve()
    _, events = _load_manifest_and_events(run_directory)
    return events


def _load_manifest_and_events(
    run_directory: Path,
) -> tuple[RunManifest, list[dict[str, Any]]]:
    """Take one locked manifest/event snapshot for a consistent status read."""
    with manifest_lock(run_directory):
        manifest = RunManifest.load(run_directory)
        expected = _event_records(manifest)
        path = event_log_path(run_directory)
        try:
            actual = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, TypeError):
            actual = []
        if actual != expected:
            try:
                write_run_event_log(manifest, run_directory)
            except OSError:
                # Projection repair is best-effort. Status is still available
                # from the authoritative manifest when this file is unwritable.
                pass
            actual = expected
        return manifest, actual


def event_log_path(run_directory: Path) -> Path:
    return run_directory / "events" / "run-events.jsonl"


def _event_records(manifest: RunManifest) -> list[dict[str, Any]]:
    ordered: list[tuple[str, int, Any]] = []
    for stage in PIPELINE_STAGE_NAMES:
        for index, event in enumerate(manifest.stages[stage].events):
            ordered.append((stage, index, event))
    ordered.sort(
        key=lambda item: (
            item[2].at,
            PIPELINE_STAGE_NAMES.index(item[0]),
            item[1],
        )
    )
    records = [
        {
            "schema_version": EVENT_LOG_SCHEMA_VERSION,
            "manifest_revision": manifest.revision,
            "sequence": 1,
            "run_id": manifest.run_id,
            "stage": None,
            "at": manifest.created_at.isoformat(),
            "event": "run_created",
            "from_status": None,
            "to_status": "created",
            "worker_id": None,
            "lease_generation": 0,
            "detail": None,
        }
    ]
    for sequence, (stage, _, event) in enumerate(ordered, start=2):
        records.append(
            {
                "schema_version": EVENT_LOG_SCHEMA_VERSION,
                "manifest_revision": manifest.revision,
                "sequence": sequence,
                "run_id": manifest.run_id,
                "stage": stage,
                "at": event.at.isoformat(),
                "event": event.event,
                "from_status": (
                    event.from_status.value if event.from_status else None
                ),
                "to_status": (
                    event.to_status.value if event.to_status else None
                ),
                "worker_id": event.worker_id,
                "lease_generation": event.lease_generation,
                "detail": (
                    redact_sensitive_text(event.detail)
                    if event.detail
                    else None
                ),
            }
        )
    return records


def _stage_detail(record: Any) -> dict[str, Any]:
    return {
        "status": record.status.value,
        "attempt_count": record.attempt_count,
        "max_attempts": record.max_attempts,
        "retryable": record.retryable,
        "next_retry_at": (
            record.next_retry_at.isoformat() if record.next_retry_at else None
        ),
        "started_at": (
            record.started_at.isoformat() if record.started_at else None
        ),
        "completed_at": (
            record.completed_at.isoformat() if record.completed_at else None
        ),
        "duration_seconds": record.duration_seconds,
        "provider": record.provider,
        "model": record.model,
        "input_fingerprint": record.input_fingerprint,
        "cost_usd": record.cost_usd,
        "resources": (
            record.resources.model_dump(mode="json")
            if record.resources
            else None
        ),
        "error": (
            {
                "class": record.error_class,
                "message": redact_sensitive_text(record.error),
            }
            if record.error
            else None
        ),
        "attempts": [
            {
                **attempt.model_dump(mode="json"),
                "error": (
                    redact_sensitive_text(attempt.error)
                    if attempt.error
                    else None
                ),
            }
            for attempt in record.attempts
        ],
        "events": [event.model_dump(mode="json") for event in record.events],
    }


def _load_work_items(
    run_directory: Path,
    *,
    pattern: str,
    kind: str,
    identifier_key: str,
) -> list[WorkItemObservation]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in sorted(run_directory.glob(pattern)):
        relative = str(path.resolve().relative_to(run_directory))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or not payload:
                raise ValueError("attempt history is empty")
            identifier = str(payload[0][identifier_key])
            attempts = [_safe_attempt(item) for item in payload]
        except (OSError, ValueError, TypeError, KeyError) as error:
            identifier = path.stem.removesuffix(".attempts")
            grouped.setdefault(
                identifier,
                {
                    "attempts": [],
                    "history_files": [],
                    "read_errors": [],
                },
            )["read_errors"].append(type(error).__name__)
            grouped[identifier]["history_files"].append(relative)
            continue
        group = grouped.setdefault(
            identifier,
            {"attempts": [], "history_files": [], "read_errors": []},
        )
        group["attempts"].extend(attempts)
        group["history_files"].append(relative)

    observations = []
    for identifier, group in grouped.items():
        attempts = sorted(
            group["attempts"], key=lambda item: item.get("started_at", "")
        )
        latest = attempts[-1] if attempts else {}
        costs = [
            item.get("cost_usd")
            for item in attempts
            if item.get("cost_usd") is not None
        ]
        observations.append(
            WorkItemObservation(
                kind=kind,
                work_item_id=identifier,
                status=(
                    str(latest.get("status", "unreadable"))
                    if not group["read_errors"] or attempts
                    else "unreadable"
                ),
                attempt_count=len(attempts),
                failed_attempts=sum(
                    item.get("status") == "failed" for item in attempts
                ),
                provider=latest.get("provider"),
                model=latest.get("model"),
                latency_seconds=sum(
                    float(item.get("latency_seconds", 0))
                    for item in attempts
                ),
                cost_usd=sum(costs) if costs else None,
                attempts=attempts,
                history_files=group["history_files"],
                read_errors=group["read_errors"],
            )
        )
    return sorted(observations, key=lambda item: item.work_item_id)


def _safe_attempt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("attempt is not an object")
    safe = dict(payload)
    if safe.get("error"):
        safe["error"] = redact_sensitive_text(str(safe["error"]))
    return safe


def _derived_errors(manifest: RunManifest) -> list[RunError]:
    errors = []
    for name, record in manifest.stages.items():
        if record.error:
            errors.append(
                RunError(
                    at=record.completed_at or manifest.updated_at,
                    stage=name,
                    error_class=record.error_class or "stage_error",
                    message=redact_sensitive_text(record.error),
                    retryable=record.status == StageStatus.QUEUED,
                    terminal=record.status == StageStatus.FAILED,
                    attempt_number=(
                        record.attempt_count
                        if record.attempt_count > 0
                        else None
                    ),
                )
            )
    return errors


def _json_list(path: str | None) -> list[Any]:
    payload = _json_value(path)
    return payload if isinstance(payload, list) else []


def _json_value(path: str | None) -> Any:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _duration_label(milliseconds: int) -> str:
    minutes, seconds = divmod(max(0, milliseconds // 1000), 60)
    return f"{minutes}:{seconds:02d}"


def _output_inside_run(run_directory: Path, path: str | None) -> bool:
    if not path:
        return False
    output = Path(path)
    if not output.is_file():
        return False
    try:
        output.resolve().relative_to(run_directory.resolve())
    except ValueError:
        return False
    return True

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


BASE_RETRY_BACKOFF_SECONDS = 30
MAX_RETRY_BACKOFF_SECONDS = 600
PIPELINE_STAGE_NAMES = (
    "ingest",
    "transcribe",
    "segment",
    "localize",
    "synthesize",
    "render",
)


class ManifestConflictError(RuntimeError):
    pass


class MutationAborted(RuntimeError):
    """Raised inside a mutate_manifest callback to skip the write."""


@dataclass(frozen=True)
class Lease:
    """Proof that a worker owns a stage.

    `lease_generation` is a fencing token: it increases on every claim, so a
    worker that stalled, lost its lease, and later woke up cannot publish state
    over the worker that reclaimed the stage.
    """

    worker_id: str
    lease_generation: int


class RunStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    INGESTED = "ingested"
    TRANSCRIBED = "transcribed"
    SEGMENTED = "segmented"
    LOCALIZED = "localized"
    SYNTHESIZED = "synthesized"
    RENDERED = "rendered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


class StageAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    status: StageStatus
    started_at: datetime
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    worker_id: str | None = None
    lease_generation: int = Field(default=0, ge=0)
    error_class: str | None = None
    error: str | None = None


class StageEvent(BaseModel):
    at: datetime
    event: str
    from_status: StageStatus | None = None
    to_status: StageStatus | None = None
    worker_id: str | None = None
    lease_generation: int = Field(default=0, ge=0)
    detail: str | None = None


class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    retryable: bool = True
    next_retry_at: datetime | None = None
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    completed_at: datetime | None = None
    worker_id: str | None = None
    lease_generation: int = Field(default=0, ge=0)
    provider: str | None = None
    model: str | None = None
    input_fingerprint: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    outputs: dict[str, str] = Field(default_factory=dict)
    error_class: str | None = None
    error: str | None = None
    attempts: list[StageAttempt] = Field(default_factory=list)
    events: list[StageEvent] = Field(default_factory=list)


class MediaMetadata(BaseModel):
    duration_seconds: float
    format_name: str | None = None
    video_codec: str
    width: int
    height: int
    frame_rate: str | None = None
    audio_codec: str
    audio_channels: int | None = None
    audio_sample_rate: int | None = None


class RunManifest(BaseModel):
    schema_version: int = 1
    revision: int = Field(default=0, ge=0)
    run_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_path: str
    source_start_ms: int
    source_end_ms: int
    source_language: str = "en"
    target_language: str = "hi"
    status: RunStatus = RunStatus.CREATED
    media: MediaMetadata | None = None
    models: dict[str, str] = Field(default_factory=dict)
    stages: dict[str, StageRecord] = Field(
        default_factory=lambda: {
            name: StageRecord() for name in PIPELINE_STAGE_NAMES
        }
    )
    outputs: dict[str, str] = Field(default_factory=dict)
    timings_seconds: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_pipeline_stages(self) -> "RunManifest":
        """Add new stages without breaking manifests created by older builds."""
        original = self.stages
        segment_was_missing = "segment" not in original
        migrated_segment = StageRecord()
        if segment_was_missing and any(
            original.get(name, StageRecord()).status != StageStatus.PENDING
            for name in ("localize", "synthesize", "render")
        ):
            compatibility_outputs = {}
            if segments_path := self.outputs.get("segments"):
                compatibility_outputs["translation_segments"] = segments_path
            migrated_segment = StageRecord(
                status=StageStatus.COMPLETED,
                completed_at=original.get("transcribe", StageRecord()).completed_at,
                outputs=compatibility_outputs,
            )
            self.outputs.update(compatibility_outputs)

        self.stages = {
            name: (
                migrated_segment
                if name == "segment" and segment_was_missing
                else original.get(name, StageRecord())
            )
            for name in PIPELINE_STAGE_NAMES
        } | {
            name: record
            for name, record in original.items()
            if name not in PIPELINE_STAGE_NAMES
        }
        return self

    @property
    def duration_ms(self) -> int:
        return self.source_end_ms - self.source_start_ms

    def save(self, run_directory: Path) -> Path:
        with manifest_lock(run_directory):
            return self._write_locked(run_directory)

    def _write_locked(self, run_directory: Path) -> Path:
        """Write the manifest. The caller must already hold the manifest lock."""
        manifest_path = run_directory / "manifest.json"
        temporary_path = run_directory / ".manifest.json.tmp"
        current_revision = _manifest_revision(manifest_path)
        if current_revision is not None and current_revision != self.revision:
            raise ManifestConflictError(
                "Manifest changed since it was loaded: "
                f"expected revision {self.revision}, "
                f"found {current_revision}."
            )

        previous_revision = self.revision
        previous_updated_at = self.updated_at
        self.revision += 1
        self.updated_at = datetime.now(timezone.utc)
        payload = self.model_dump(mode="json")

        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, manifest_path)
        except Exception:
            self.revision = previous_revision
            self.updated_at = previous_updated_at
            raise
        return manifest_path

    @classmethod
    def load(cls, run_directory: Path) -> "RunManifest":
        manifest_path = run_directory / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))

    def public_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "source": self.source_path,
            "range_ms": [self.source_start_ms, self.source_end_ms],
            "source_language": self.source_language,
            "target_language": self.target_language,
            "stages": {
                name: stage.status.value for name, stage in self.stages.items()
            },
            "outputs": self.outputs,
            "errors": self.errors,
        }


@contextmanager
def manifest_lock(run_directory: Path) -> Generator[None, None, None]:
    """Hold the exclusive run lock.

    `flock` is advisory and process-local to POSIX filesystems, so this only
    provides compare-and-set safety for workers sharing a local disk. Network
    filesystems and remote deployments need a state store with real conditional
    writes instead.
    """
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = run_directory / ".manifest.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def mutate_manifest(
    run_directory: Path,
    apply: Callable[["RunManifest"], None],
) -> RunManifest:
    """Load, mutate, and write a manifest inside one exclusive lock.

    Long-running work must never hold a manifest across its own execution: the
    revision check would reject the final write as soon as anything else
    (a heartbeat, a cancellation) touched the run. Keep every read-modify-write
    inside this short critical section instead.

    Raise `MutationAborted` from `apply` to leave the manifest untouched.
    """
    with manifest_lock(run_directory):
        manifest = RunManifest.load(run_directory)
        try:
            apply(manifest)
        except MutationAborted:
            return manifest
        manifest._write_locked(run_directory)
        return manifest


def retry_delay_seconds(attempt_number: int) -> float:
    """Exponential backoff, capped, before the next attempt of a failed stage."""
    exponent = max(0, attempt_number - 1)
    delay = BASE_RETRY_BACKOFF_SECONDS * (2**exponent)
    return float(min(delay, MAX_RETRY_BACKOFF_SECONDS))


def append_stage_event(
    record: StageRecord,
    *,
    at: datetime,
    event: str,
    from_status: StageStatus | None = None,
    to_status: StageStatus | None = None,
    worker_id: str | None = None,
    lease_generation: int = 0,
    detail: str | None = None,
) -> None:
    record.events.append(
        StageEvent(
            at=at,
            event=event,
            from_status=from_status,
            to_status=to_status,
            worker_id=worker_id,
            lease_generation=lease_generation,
            detail=detail,
        )
    )


def stage_holds_lease(
    record: StageRecord | None,
    lease: Lease | None,
    *,
    now: datetime | None = None,
) -> bool:
    """Check that a lease still owns a stage.

    An absent lease means the caller is a trusted direct invocation (the CLI or
    the web app in local mode) rather than a leased worker.
    """
    if record is None:
        return False
    if lease is None:
        return not (
            record.status == StageStatus.RUNNING
            and record.worker_id is not None
        )
    moment = now or datetime.now(timezone.utc)
    return (
        record.status == StageStatus.RUNNING
        and record.worker_id == lease.worker_id
        and record.lease_generation == lease.lease_generation
        and record.lease_expires_at is not None
        and record.lease_expires_at > moment
    )


def begin_stage(
    run_directory: Path,
    stage: str,
    *,
    lease: Lease | None = None,
    now: datetime | None = None,
) -> RunManifest | None:
    """Mark a stage running. Returns None when a stale lease is fenced out."""
    moment = now or datetime.now(timezone.utc)
    accepted: list[RunManifest] = []

    def apply(manifest: RunManifest) -> None:
        record = manifest.stages.setdefault(stage, StageRecord())
        if not stage_holds_lease(record, lease, now=moment):
            raise MutationAborted
        accepted.append(manifest)
        if record.status == StageStatus.RUNNING:
            return
        previous_status = record.status
        record.status = StageStatus.RUNNING
        record.started_at = moment
        record.heartbeat_at = moment
        record.completed_at = None
        record.error = None
        record.error_class = None
        manifest.status = RunStatus.RUNNING
        append_stage_event(
            record,
            at=moment,
            event="started",
            from_status=previous_status,
            to_status=StageStatus.RUNNING,
        )

    manifest = mutate_manifest(run_directory, apply)
    return manifest if accepted else None


def renew_lease(
    run_directory: Path,
    stage: str,
    *,
    lease: Lease,
    lease_seconds: int,
    now: datetime | None = None,
) -> bool:
    """Extend a lease. Returns False once the lease has been taken over."""
    if lease_seconds <= 0:
        raise ValueError("Lease seconds must be greater than zero.")
    moment = now or datetime.now(timezone.utc)
    renewed: list[bool] = []

    def apply(manifest: RunManifest) -> None:
        record = manifest.stages.get(stage)
        if not stage_holds_lease(record, lease, now=moment):
            raise MutationAborted
        assert record is not None
        record.heartbeat_at = moment
        record.lease_expires_at = moment + timedelta(seconds=lease_seconds)
        if record.attempts:
            record.attempts[-1].heartbeat_at = moment
        append_stage_event(
            record,
            at=moment,
            event="heartbeat",
            from_status=StageStatus.RUNNING,
            to_status=StageStatus.RUNNING,
            worker_id=lease.worker_id,
            lease_generation=lease.lease_generation,
        )
        renewed.append(True)

    mutate_manifest(run_directory, apply)
    return bool(renewed)


def complete_stage(
    run_directory: Path,
    stage: str,
    *,
    lease: Lease | None = None,
    outputs: dict[str, str] | None = None,
    run_status: RunStatus | None = None,
    models: dict[str, str] | None = None,
    media: MediaMetadata | None = None,
    duration_seconds: float | None = None,
    provider: str | None = None,
    input_fingerprint: str | None = None,
    cost_usd: float | None = None,
    record_cost: bool = False,
    now: datetime | None = None,
) -> RunManifest | None:
    """Commit a successful stage. Returns None when a stale lease is fenced out."""
    moment = now or datetime.now(timezone.utc)
    accepted: list[RunManifest] = []

    def apply(manifest: RunManifest) -> None:
        record = manifest.stages.get(stage)
        if not stage_holds_lease(record, lease, now=moment):
            raise MutationAborted
        assert record is not None
        previous_status = record.status
        record.status = StageStatus.COMPLETED
        record.completed_at = moment
        record.heartbeat_at = moment
        record.lease_expires_at = None
        record.worker_id = None
        record.next_retry_at = None
        record.error = None
        record.error_class = None
        if outputs:
            record.outputs = outputs
            manifest.outputs.update(outputs)
        if media is not None:
            manifest.media = media
        if duration_seconds is not None:
            record.duration_seconds = duration_seconds
            manifest.timings_seconds[stage] = duration_seconds
        if provider is not None:
            record.provider = provider
        if input_fingerprint is not None:
            record.input_fingerprint = input_fingerprint
        if record_cost:
            record.cost_usd = cost_usd
        if models:
            manifest.models.update(models)
            record.model = next(iter(models.values()))
        if run_status is not None:
            manifest.status = run_status
        _close_current_attempt(record, StageStatus.COMPLETED, moment)
        append_stage_event(
            record,
            at=moment,
            event="completed",
            from_status=previous_status,
            to_status=StageStatus.COMPLETED,
            worker_id=lease.worker_id if lease else None,
            lease_generation=lease.lease_generation if lease else 0,
        )
        accepted.append(manifest)

    manifest = mutate_manifest(run_directory, apply)
    return manifest if accepted else None


def fail_stage(
    run_directory: Path,
    stage: str,
    *,
    error: str,
    lease: Lease | None = None,
    error_class: str | None = None,
    retryable: bool = False,
    retry_delay_seconds: float = 0.0,
    now: datetime | None = None,
) -> RunManifest | None:
    """Record a stage failure, queueing a bounded retry when one is allowed."""
    moment = now or datetime.now(timezone.utc)
    accepted: list[RunManifest] = []

    def apply(manifest: RunManifest) -> None:
        record = manifest.stages.get(stage)
        if not stage_holds_lease(record, lease, now=moment):
            raise MutationAborted
        assert record is not None
        previous_status = record.status
        record.completed_at = moment
        record.lease_expires_at = None
        record.worker_id = None
        record.error = error
        record.error_class = error_class
        if (
            retryable
            and record.retryable
            and record.attempt_count < record.max_attempts
        ):
            record.status = StageStatus.QUEUED
            record.next_retry_at = moment + timedelta(seconds=retry_delay_seconds)
            manifest.status = RunStatus.QUEUED
        else:
            record.status = StageStatus.FAILED
            record.retryable = False
            record.next_retry_at = None
            manifest.status = RunStatus.FAILED
        manifest.errors.append(error)
        _close_current_attempt(
            record,
            StageStatus.FAILED,
            moment,
            error_class=error_class,
            error=error,
        )
        append_stage_event(
            record,
            at=moment,
            event="retry_queued" if record.status == StageStatus.QUEUED else "failed",
            from_status=previous_status,
            to_status=record.status,
            worker_id=lease.worker_id if lease else None,
            lease_generation=lease.lease_generation if lease else 0,
            detail=error,
        )
        accepted.append(manifest)

    manifest = mutate_manifest(run_directory, apply)
    return manifest if accepted else None


def _close_current_attempt(
    record: StageRecord,
    status: StageStatus,
    moment: datetime,
    *,
    error_class: str | None = None,
    error: str | None = None,
) -> None:
    if not record.attempts:
        return
    attempt = record.attempts[-1]
    attempt.status = status
    attempt.completed_at = moment
    attempt.heartbeat_at = moment
    attempt.error_class = error_class
    attempt.error = error


def _manifest_revision(manifest_path: Path) -> int | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return int(payload.get("revision", 0))

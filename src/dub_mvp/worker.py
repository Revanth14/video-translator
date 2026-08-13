from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dub_mvp.manifest import (
    Lease,
    MutationAborted,
    RunManifest,
    RunStatus,
    StageAttempt,
    StageRecord,
    StageStatus,
    append_run_error,
    append_stage_event,
    mutate_manifest,
    renew_lease,
)
from dub_mvp.runner import JobRunner, LocalJobRunner, StageRequest


LOGGER = logging.getLogger(__name__)

STAGE_ORDER = [
    "ingest",
    "transcribe",
    "segment",
    "localize",
    "synthesize",
    "render",
]
MAX_LOOP_BACKOFF_SECONDS = 30.0
DEFAULT_LEASE_SECONDS = 120
DEFAULT_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueuedJob:
    run_directory: Path
    run_id: str
    stage: str


@dataclass(frozen=True)
class WorkerResult:
    processed: bool
    run_id: str | None = None
    stage: str | None = None
    status: str | None = None


def is_claimable(record: StageRecord | None, moment: datetime) -> bool:
    """Decide whether a stage is eligible for a worker to pick up.

    Queued work becomes eligible once its retry backoff has elapsed. Running
    work becomes eligible again once its lease expires, which is how a stage
    abandoned by a dead worker gets reclaimed.
    """
    if record is None:
        return False
    if record.status == StageStatus.QUEUED:
        return record.next_retry_at is None or record.next_retry_at <= moment
    if record.status == StageStatus.RUNNING:
        return (
            record.lease_expires_at is not None
            and record.lease_expires_at <= moment
        )
    return False


def advance_and_find_job(
    runs_directory: Path,
    *,
    now: datetime | None = None,
    exclude: set[Path] | None = None,
) -> QueuedJob | None:
    """Advance every run's progression, then return the first claimable stage.

    This writes as well as reads: queueing the next ready stage is what makes
    progression a property of durable state rather than of whoever happens to
    call in. The name says so.
    """
    moment = now or datetime.now(timezone.utc)
    skipped = exclude or set()
    manifests = sorted(
        runs_directory.expanduser().resolve().glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    for manifest_path in manifests:
        if manifest_path.parent in skipped:
            continue
        job = _scan_run(manifest_path.parent, moment=moment)
        if job is not None:
            return job
    return None


def _scan_run(
    run_directory: Path,
    *,
    moment: datetime,
) -> QueuedJob | None:
    """Advance and inspect one run, or return None if it cannot be read.

    Faults are contained per run so that a single unreadable or unwritable run
    directory cannot stop the worker from serving every other run.
    """
    try:
        manifest = RunManifest.load(run_directory)
        queue_next_ready_stage(run_directory, manifest=manifest)
        manifest = RunManifest.load(run_directory)
    except (OSError, ValueError) as error:
        LOGGER.warning("Skipping unreadable run %s: %s", run_directory, error)
        return None

    for stage_name in STAGE_ORDER:
        if _dependencies_complete(manifest, stage_name) and is_claimable(
            manifest.stages.get(stage_name), moment
        ):
            return QueuedJob(
                run_directory=run_directory,
                run_id=manifest.run_id,
                stage=stage_name,
            )
    return None


def queue_next_ready_stage(
    run_directory: Path,
    *,
    manifest: RunManifest | None = None,
) -> str | None:
    """Queue the first pending stage whose dependency is complete.

    This makes progression a property of durable state. It is safe to call on
    every scan because the mutation re-checks state under the manifest lock.
    """
    snapshot = manifest or RunManifest.load(run_directory)
    candidate = _next_ready_stage(snapshot)
    if candidate is None:
        return None
    queued: list[str] = []

    def apply(current: RunManifest) -> None:
        if _next_ready_stage(current) != candidate:
            raise MutationAborted
        record = current.stages[candidate]
        record.status = StageStatus.QUEUED
        record.next_retry_at = None
        record.completed_at = None
        record.error = None
        record.error_class = None
        current.status = RunStatus.QUEUED
        append_stage_event(
            record,
            at=datetime.now(timezone.utc),
            event="auto_queued",
            from_status=StageStatus.PENDING,
            to_status=StageStatus.QUEUED,
        )
        queued.append(candidate)

    mutate_manifest(run_directory, apply)
    return queued[0] if queued else None


def _next_ready_stage(manifest: RunManifest) -> str | None:
    if manifest.status in {
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.RENDERED,
    }:
        return None
    if any(
        record.status in {
            StageStatus.QUEUED,
            StageStatus.RUNNING,
            StageStatus.FAILED,
            StageStatus.CANCELLED,
        }
        for record in manifest.stages.values()
    ):
        return None
    for index, stage in enumerate(STAGE_ORDER):
        record = manifest.stages.get(stage)
        if record is None:
            return None
        if record.status in {StageStatus.PENDING, StageStatus.INVALIDATED}:
            if index == 0:
                return stage
            previous = manifest.stages.get(STAGE_ORDER[index - 1])
            return (
                stage
                if previous is not None
                and previous.status == StageStatus.COMPLETED
                else None
            )
    return None


def _dependencies_complete(manifest: RunManifest, stage: str) -> bool:
    try:
        index = STAGE_ORDER.index(stage)
    except ValueError:
        return False
    return all(
        manifest.stages.get(previous) is not None
        and manifest.stages[previous].status == StageStatus.COMPLETED
        for previous in STAGE_ORDER[:index]
    )


def run_worker_once(
    *,
    runs_directory: Path,
    runner: JobRunner | None = None,
    worker_id: str = DEFAULT_WORKER_ID,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> WorkerResult:
    active_runner = runner or LocalJobRunner(background=False)
    # A run that cannot be claimed or written this pass is set aside so it
    # cannot block the runs behind it. The set is per-pass, so a run that
    # recovers is picked up on the next scan.
    unavailable: set[Path] = set()

    while True:
        job = advance_and_find_job(runs_directory, exclude=unavailable)
        if job is None:
            return WorkerResult(processed=False)

        try:
            lease = claim_job(
                job,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        except (OSError, ValueError) as error:
            LOGGER.warning(
                "Cannot claim %s %s: %s", job.run_id, job.stage, error
            )
            unavailable.add(job.run_directory)
            continue

        if lease is None:
            # Claimed by another worker, or just retired. Look elsewhere.
            unavailable.add(job.run_directory)
            continue

        try:
            with LeaseKeeper(
                run_directory=job.run_directory,
                stage=job.stage,
                lease=lease,
                lease_seconds=lease_seconds,
            ):
                active_runner.submit_stage(_stage_request(job, lease))

            manifest = RunManifest.load(job.run_directory)
            if manifest.stages[job.stage].status == StageStatus.COMPLETED:
                queue_next_ready_stage(job.run_directory, manifest=manifest)
                manifest = RunManifest.load(job.run_directory)
        except (OSError, ValueError) as error:
            # Stage outcomes are recorded by the runner; reaching here means
            # the run's own state could not be written. Leave the lease to
            # expire so the stage is reclaimed once the fault clears.
            LOGGER.warning(
                "Cannot record %s %s: %s", job.run_id, job.stage, error
            )
            unavailable.add(job.run_directory)
            continue

        return WorkerResult(
            processed=True,
            run_id=job.run_id,
            stage=job.stage,
            status=manifest.stages[job.stage].status.value,
        )


def run_worker_loop(
    *,
    runs_directory: Path,
    poll_seconds: float,
    runner: JobRunner | None = None,
    max_iterations: int | None = None,
) -> None:
    """Serve queued work until stopped.

    The loop is the supervisor for every run in this process, so it must
    outlive any single failure. An unguarded exception here would silently
    strand every job: the web app keeps answering requests while nothing ever
    progresses again.
    """
    if poll_seconds <= 0:
        raise WorkerError("Poll seconds must be greater than zero.")
    consecutive_failures = 0
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            result = run_worker_once(
                runs_directory=runs_directory,
                runner=runner,
            )
        except Exception:  # noqa: BLE001 - the supervisor must never exit
            consecutive_failures += 1
            LOGGER.exception(
                "Worker iteration failed (%s consecutive); backing off.",
                consecutive_failures,
            )
            time.sleep(_failure_backoff(poll_seconds, consecutive_failures))
            continue
        consecutive_failures = 0
        if not result.processed:
            time.sleep(poll_seconds)


def _failure_backoff(poll_seconds: float, consecutive_failures: int) -> float:
    """Back off on repeated faults so a persistent one cannot spin hot."""
    exponent = min(consecutive_failures - 1, 6)
    return min(poll_seconds * (2**exponent), MAX_LOOP_BACKOFF_SECONDS)


def claim_job(
    job: QueuedJob,
    *,
    worker_id: str = DEFAULT_WORKER_ID,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> Lease | None:
    """Claim a stage and return its lease, or None when it cannot be claimed.

    A stage that has exhausted its attempts is moved to a terminal failure here
    rather than left queued: leaving it claimable would spin the worker loop
    forever and starve every other run behind it.
    """
    if lease_seconds <= 0:
        raise WorkerError("Lease seconds must be greater than zero.")
    claimed_at = now or datetime.now(timezone.utc)
    granted: list[Lease] = []

    def apply(manifest: RunManifest) -> None:
        record = manifest.stages.get(job.stage)
        if not is_claimable(record, claimed_at):
            raise MutationAborted
        assert record is not None
        if record.status == StageStatus.RUNNING:
            _abandon_expired_attempt(record, claimed_at)
        if record.attempt_count >= record.max_attempts:
            _exhaust_attempts(manifest, job.stage, record, claimed_at)
            return
        record.status = StageStatus.RUNNING
        record.attempt_count += 1
        record.started_at = claimed_at
        record.heartbeat_at = claimed_at
        record.lease_expires_at = claimed_at + timedelta(seconds=lease_seconds)
        record.completed_at = None
        record.worker_id = worker_id
        record.lease_generation += 1
        record.next_retry_at = None
        record.error_class = None
        record.error = None
        record.attempts.append(
            StageAttempt(
                attempt_number=record.attempt_count,
                status=StageStatus.RUNNING,
                started_at=claimed_at,
                heartbeat_at=claimed_at,
                worker_id=worker_id,
                lease_generation=record.lease_generation,
            )
        )
        append_stage_event(
            record,
            at=claimed_at,
            event="claimed",
            from_status=(
                StageStatus.RUNNING
                if record.attempt_count > 1 and record.attempts[-2].error_class == "lease_expired"
                else StageStatus.QUEUED
            ),
            to_status=StageStatus.RUNNING,
            worker_id=worker_id,
            lease_generation=record.lease_generation,
        )
        manifest.status = RunStatus.RUNNING
        granted.append(
            Lease(worker_id=worker_id, lease_generation=record.lease_generation)
        )

    mutate_manifest(job.run_directory, apply)
    return granted[0] if granted else None


def _exhaust_attempts(
    manifest: RunManifest,
    stage: str,
    record: StageRecord,
    moment: datetime,
) -> None:
    message = (
        f"Stage {stage} exhausted {record.max_attempts} attempts and will not "
        "be retried automatically."
    )
    record.status = StageStatus.FAILED
    record.retryable = False
    record.error_class = "attempts_exhausted"
    record.error = message
    record.completed_at = moment
    record.next_retry_at = None
    record.worker_id = None
    record.lease_expires_at = None
    manifest.status = RunStatus.FAILED
    append_run_error(
        manifest,
        at=moment,
        stage=stage,
        error_class="attempts_exhausted",
        message=message,
        retryable=False,
        terminal=True,
        attempt_number=(
            record.attempt_count if record.attempt_count > 0 else None
        ),
    )
    append_stage_event(
        record,
        at=moment,
        event="attempts_exhausted",
        from_status=StageStatus.QUEUED,
        to_status=StageStatus.FAILED,
        detail=message,
    )


def _abandon_expired_attempt(record: StageRecord, moment: datetime) -> None:
    if not record.attempts:
        return
    attempt = record.attempts[-1]
    if attempt.status != StageStatus.RUNNING:
        return
    attempt.status = StageStatus.FAILED
    attempt.completed_at = moment
    attempt.error_class = "lease_expired"
    attempt.error = (
        f"Lease held by {attempt.worker_id or 'unknown worker'} expired "
        "and the stage was reclaimed."
    )
    append_stage_event(
        record,
        at=moment,
        event="lease_expired",
        from_status=StageStatus.RUNNING,
        to_status=StageStatus.FAILED,
        worker_id=attempt.worker_id,
        lease_generation=attempt.lease_generation,
        detail=attempt.error,
    )


class LeaseKeeper:
    """Renew a stage lease in the background while the stage executes.

    Without this a long stage such as transcription would look abandoned to
    other workers and be reclaimed mid-flight.
    """

    def __init__(
        self,
        *,
        run_directory: Path,
        stage: str,
        lease: Lease,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self.run_directory = run_directory
        self.stage = stage
        self.lease = lease
        self.lease_seconds = lease_seconds
        self.interval = max(1.0, lease_seconds / 3)
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LeaseKeeper":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                renewed = renew_lease(
                    self.run_directory,
                    self.stage,
                    lease=self.lease,
                    lease_seconds=self.lease_seconds,
                )
            except (OSError, ValueError):
                continue
            if not renewed:
                self.lost.set()
                return


def _stage_request(job: QueuedJob, lease: Lease | None = None) -> StageRequest:
    input_directory = job.run_directory / "input"
    glossary_path = input_directory / "glossary.json"
    translation_context_path = input_directory / "translation-context.json"
    voice_reference_path = input_directory / "voice-reference.json"
    return StageRequest(
        run_directory=job.run_directory,
        stage=job.stage,
        glossary_path=(
            glossary_path if job.stage == "localize" and glossary_path.exists()
            else None
        ),
        translation_context_path=(
            translation_context_path
            if job.stage == "localize" and translation_context_path.exists()
            else None
        ),
        voice_reference_path=(
            voice_reference_path
            if job.stage == "synthesize" and voice_reference_path.exists()
            else None
        ),
        lease=lease,
    )

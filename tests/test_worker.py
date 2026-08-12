import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Thread

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import (
    RunManifest,
    RunStatus,
    StageStatus,
    complete_stage,
    begin_stage,
    fail_stage,
    renew_lease,
    retry_delay_seconds,
)
from dub_mvp.runner import StageRequest
from dub_mvp.runner import QueuedJobRunner
from dub_mvp import worker
from dub_mvp.worker import (
    QueuedJob,
    WorkerResult,
    claim_job,
    advance_and_find_job,
    run_worker_loop,
    run_worker_once,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.ingested: list[Path] = []
        self.stages: list[StageRequest] = []

    def submit_ingest(self, run_directory: Path) -> None:
        self.ingested.append(run_directory)
        manifest = RunManifest.load(run_directory)
        record = manifest.stages["ingest"]
        record.status = StageStatus.COMPLETED
        manifest.status = RunStatus.INGESTED
        manifest.save(run_directory)

    def submit_stage(self, request: StageRequest) -> None:
        self.stages.append(request)
        if request.stage == "ingest":
            self.ingested.append(request.run_directory)
        manifest = RunManifest.load(request.run_directory)
        record = manifest.stages[request.stage]
        record.status = StageStatus.COMPLETED
        manifest.status = {
            "ingest": RunStatus.INGESTED,
            "transcribe": RunStatus.TRANSCRIBED,
            "segment": RunStatus.SEGMENTED,
            "localize": RunStatus.LOCALIZED,
            "synthesize": RunStatus.SYNTHESIZED,
            "render": RunStatus.RENDERED,
        }[request.stage]
        manifest.save(request.run_directory)


def write_manifest(
    run_directory: Path,
    *,
    run_id: str,
    queued_stage: str | None = None,
) -> RunManifest:
    manifest = RunManifest(
        run_id=run_id,
        source_path=str(run_directory / "input.mp4"),
        source_start_ms=0,
        source_end_ms=90000,
    )
    if queued_stage:
        queued_index = [
            "ingest",
            "transcribe",
            "segment",
            "localize",
            "synthesize",
            "render",
        ].index(queued_stage)
        for completed_stage in [
            "ingest",
            "transcribe",
            "segment",
            "localize",
            "synthesize",
            "render",
        ][:queued_index]:
            manifest.stages[completed_stage].status = StageStatus.COMPLETED
        manifest.status = RunStatus.QUEUED
        manifest.stages[queued_stage].status = StageStatus.QUEUED
    manifest.save(run_directory)
    return manifest


def test_advance_and_find_job_uses_stage_order(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    manifest = write_manifest(run_directory, run_id="run-a")
    manifest.status = RunStatus.QUEUED
    manifest.stages["ingest"].status = StageStatus.COMPLETED
    manifest.stages["render"].status = StageStatus.QUEUED
    manifest.stages["transcribe"].status = StageStatus.QUEUED
    manifest.save(run_directory)

    job = advance_and_find_job(tmp_path)

    assert job == QueuedJob(
        run_directory=run_directory,
        run_id="run-a",
        stage="transcribe",
    )


def test_claim_job_marks_stage_running(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")

    claimed_at = datetime(2026, 8, 12, tzinfo=timezone.utc)
    assert claim_job(
        QueuedJob(run_directory=run_directory, run_id="run-a", stage="ingest"),
        worker_id="worker-a",
        lease_seconds=60,
        now=claimed_at,
    )
    manifest = RunManifest.load(run_directory)

    assert manifest.status == RunStatus.RUNNING
    assert manifest.stages["ingest"].status == StageStatus.RUNNING
    assert manifest.stages["ingest"].started_at == claimed_at
    assert manifest.stages["ingest"].heartbeat_at == claimed_at
    assert manifest.stages["ingest"].worker_id == "worker-a"
    assert manifest.stages["ingest"].lease_generation == 1
    assert manifest.stages["ingest"].attempt_count == 1
    assert manifest.stages["ingest"].attempts[0].status == StageStatus.RUNNING


def test_claim_job_retires_a_stage_that_exhausted_its_attempts(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run-a"
    manifest = write_manifest(
        run_directory,
        run_id="run-a",
        queued_stage="ingest",
    )
    manifest.stages["ingest"].attempt_count = 3
    manifest.stages["ingest"].max_attempts = 3
    manifest.save(run_directory)

    assert (
        claim_job(
            QueuedJob(
                run_directory=run_directory,
                run_id="run-a",
                stage="ingest",
            )
        )
        is None
    )

    reloaded = RunManifest.load(run_directory)
    record = reloaded.stages["ingest"]
    assert record.status == StageStatus.FAILED
    assert record.error_class == "attempts_exhausted"
    assert not record.retryable
    assert reloaded.status == RunStatus.FAILED
    # A terminal stage must stop being claimable, otherwise the worker loop
    # spins on it forever and starves every other run.
    assert advance_and_find_job(tmp_path) is None


def test_expired_lease_is_reclaimed_by_another_worker(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="transcribe")
    job = QueuedJob(
        run_directory=run_directory,
        run_id="run-a",
        stage="transcribe",
    )
    start = datetime(2026, 8, 12, tzinfo=timezone.utc)

    first = claim_job(job, worker_id="worker-a", lease_seconds=60, now=start)
    assert first is not None

    # A live lease keeps the stage off the queue.
    assert advance_and_find_job(tmp_path, now=start + timedelta(seconds=30)) is None

    expired = start + timedelta(seconds=90)
    assert advance_and_find_job(tmp_path, now=expired) == job

    second = claim_job(job, worker_id="worker-b", lease_seconds=60, now=expired)
    assert second is not None
    assert second.lease_generation == first.lease_generation + 1

    record = RunManifest.load(run_directory).stages["transcribe"]
    assert record.worker_id == "worker-b"
    assert record.attempt_count == 2
    assert record.attempts[0].status == StageStatus.FAILED
    assert record.attempts[1].status == StageStatus.RUNNING
    assert record.attempts[0].error_class == "lease_expired"


def test_stale_worker_cannot_commit_after_losing_its_lease(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="transcribe")
    job = QueuedJob(
        run_directory=run_directory,
        run_id="run-a",
        stage="transcribe",
    )
    start = datetime(2026, 8, 12, tzinfo=timezone.utc)
    expired = start + timedelta(seconds=90)

    stale = claim_job(job, worker_id="worker-a", lease_seconds=60, now=start)
    current = claim_job(job, worker_id="worker-b", lease_seconds=60, now=expired)
    assert stale is not None and current is not None

    assert not renew_lease(
        run_directory,
        "transcribe",
        lease=stale,
        lease_seconds=60,
        now=expired,
    )
    assert (
        complete_stage(
            run_directory,
            "transcribe",
            lease=stale,
            outputs={"transcript": "stale.json"},
            run_status=RunStatus.TRANSCRIBED,
            now=expired,
        )
        is None
    )
    assert (
        fail_stage(
            run_directory,
            "transcribe",
            error="stale failure",
            lease=stale,
            now=expired,
        )
        is None
    )

    manifest = RunManifest.load(run_directory)
    assert manifest.stages["transcribe"].status == StageStatus.RUNNING
    assert manifest.stages["transcribe"].worker_id == "worker-b"
    assert "transcript" not in manifest.outputs

    assert (
        complete_stage(
            run_directory,
            "transcribe",
            lease=current,
            outputs={"transcript": "fresh.json"},
            run_status=RunStatus.TRANSCRIBED,
            now=expired,
        )
        is not None
    )
    assert RunManifest.load(run_directory).outputs["transcript"] == "fresh.json"


def test_retryable_failure_waits_for_backoff_before_being_claimable(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="transcribe")
    job = QueuedJob(
        run_directory=run_directory,
        run_id="run-a",
        stage="transcribe",
    )
    start = datetime(2026, 8, 12, tzinfo=timezone.utc)

    lease = claim_job(job, worker_id="worker-a", lease_seconds=60, now=start)
    assert lease is not None
    fail_stage(
        run_directory,
        "transcribe",
        error="provider timeout",
        lease=lease,
        error_class="TranscriptionError",
        retryable=True,
        retry_delay_seconds=retry_delay_seconds(1),
        now=start,
    )

    record = RunManifest.load(run_directory).stages["transcribe"]
    assert record.status == StageStatus.QUEUED
    assert record.attempts[0].status == StageStatus.FAILED
    assert record.next_retry_at == start + timedelta(seconds=30)

    assert advance_and_find_job(tmp_path, now=start + timedelta(seconds=10)) is None
    assert advance_and_find_job(tmp_path, now=start + timedelta(seconds=31)) == job


def test_heartbeat_renews_record_and_attempt(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")
    job = QueuedJob(run_directory=run_directory, run_id="run-a", stage="ingest")
    start = datetime(2026, 8, 12, tzinfo=timezone.utc)
    lease = claim_job(job, worker_id="worker-a", lease_seconds=60, now=start)
    assert lease is not None
    heartbeat = start + timedelta(seconds=30)

    assert renew_lease(
        run_directory,
        "ingest",
        lease=lease,
        lease_seconds=60,
        now=heartbeat,
    )

    record = RunManifest.load(run_directory).stages["ingest"]
    assert record.heartbeat_at == heartbeat
    assert record.lease_expires_at == heartbeat + timedelta(seconds=60)
    assert record.attempts[0].heartbeat_at == heartbeat
    assert record.attempt_count == 1


def test_unleased_stage_transition_cannot_bypass_active_lease(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")
    job = QueuedJob(run_directory=run_directory, run_id="run-a", stage="ingest")
    lease = claim_job(job, worker_id="worker-a")
    assert lease is not None
    revision = RunManifest.load(run_directory).revision

    assert begin_stage(run_directory, "ingest") is None
    assert (
        complete_stage(
            run_directory,
            "ingest",
            outputs={"probe": "unleased.json"},
            run_status=RunStatus.INGESTED,
        )
        is None
    )

    manifest = RunManifest.load(run_directory)
    assert manifest.revision == revision
    assert manifest.stages["ingest"].status == StageStatus.RUNNING
    assert "probe" not in manifest.outputs


def test_worker_loop_survives_a_failing_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loop supervises every run in the process. If one iteration's fault
    # escaped, every job would be stranded with the web app still serving.
    attempts: list[int] = []

    def flaky(**_: object) -> WorkerResult:
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("disk temporarily unavailable")
        return WorkerResult(processed=False)

    monkeypatch.setattr(worker, "run_worker_once", flaky)

    run_worker_loop(runs_directory=tmp_path, poll_seconds=0.01, max_iterations=4)

    assert len(attempts) == 4


def test_unwritable_run_does_not_starve_the_runs_behind_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "run-blocked"
    healthy = tmp_path / "run-healthy"
    write_manifest(blocked, run_id="run-blocked", queued_stage="ingest")
    write_manifest(healthy, run_id="run-healthy", queued_stage="ingest")
    # Scan order is by mtime, so the blocked run is reached first.
    os.utime(blocked / "manifest.json", (1, 1))
    os.utime(healthy / "manifest.json", (2, 2))

    real_claim = worker.claim_job

    def flaky_claim(job: QueuedJob, **kwargs: object):
        if job.run_id == "run-blocked":
            raise OSError("read-only file system")
        return real_claim(job, **kwargs)

    monkeypatch.setattr(worker, "claim_job", flaky_claim)
    runner = RecordingRunner()

    result = run_worker_once(runs_directory=tmp_path, runner=runner)

    assert result.processed
    assert result.run_id == "run-healthy"


def test_worker_once_processes_queued_ingest(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")
    runner = RecordingRunner()

    result = run_worker_once(runs_directory=tmp_path, runner=runner)

    assert result.processed
    assert result.run_id == "run-a"
    assert result.stage == "ingest"
    assert result.status == "completed"
    assert runner.ingested == [run_directory]


def test_worker_once_processes_queued_stage_with_inputs(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="synthesize")
    voice_reference = run_directory / "input" / "voice-reference.json"
    voice_reference.parent.mkdir(parents=True)
    voice_reference.write_text('{"path":null}\n', encoding="utf-8")
    runner = RecordingRunner()

    result = run_worker_once(runs_directory=tmp_path, runner=runner)

    assert result.processed
    assert result.stage == "synthesize"
    assert runner.stages[0].stage == "synthesize"
    assert runner.stages[0].voice_reference_path == voice_reference


def test_worker_automatically_progresses_through_all_stages(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a")
    voice_reference = run_directory / "input" / "voice-reference.json"
    voice_reference.parent.mkdir(parents=True)
    voice_reference.write_text('{"path":null}\n', encoding="utf-8")
    QueuedJobRunner().submit_ingest(run_directory)
    runner = RecordingRunner()

    results = [
        run_worker_once(runs_directory=tmp_path, runner=runner)
        for _ in range(6)
    ]
    manifest = RunManifest.load(run_directory)

    assert [result.stage for result in results] == [
        "ingest",
        "transcribe",
        "segment",
        "localize",
        "synthesize",
        "render",
    ]
    assert all(
        record.status == StageStatus.COMPLETED
        for record in manifest.stages.values()
    )
    assert manifest.status == RunStatus.RENDERED
    assert not run_worker_once(
        runs_directory=tmp_path,
        runner=runner,
    ).processed


def test_concurrent_claims_grant_only_one_lease(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")
    job = QueuedJob(
        run_directory=run_directory,
        run_id="run-a",
        stage="ingest",
    )
    barrier = Barrier(3)
    leases = []

    def claim(worker_id: str) -> None:
        barrier.wait()
        leases.append(claim_job(job, worker_id=worker_id))

    threads = [
        Thread(target=claim, args=("worker-a",)),
        Thread(target=claim, args=("worker-b",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sum(lease is not None for lease in leases) == 1
    record = RunManifest.load(run_directory).stages["ingest"]
    assert record.attempt_count == 1
    assert len(record.attempts) == 1


def test_worker_once_reports_empty_queue(tmp_path: Path) -> None:
    result = run_worker_once(runs_directory=tmp_path, runner=RecordingRunner())

    assert not result.processed


def test_worker_command_once_reports_empty_queue(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["worker", "--runs", str(tmp_path), "--once"],
    )

    assert result.exit_code == 0
    assert "No queued jobs." in result.output

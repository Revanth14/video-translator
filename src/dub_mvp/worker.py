from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.runner import JobRunner, LocalJobRunner, StageRequest


STAGE_ORDER = ["ingest", "transcribe", "localize", "synthesize", "render"]


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


def find_next_queued_job(runs_directory: Path) -> QueuedJob | None:
    manifests = sorted(
        runs_directory.expanduser().resolve().glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    for manifest_path in manifests:
        try:
            manifest = RunManifest.load(manifest_path.parent)
        except (OSError, ValueError):
            continue
        for stage_name in STAGE_ORDER:
            record = manifest.stages.get(stage_name)
            if record and record.status == StageStatus.QUEUED:
                return QueuedJob(
                    run_directory=manifest_path.parent,
                    run_id=manifest.run_id,
                    stage=stage_name,
                )
    return None


def run_worker_once(
    *,
    runs_directory: Path,
    runner: JobRunner | None = None,
) -> WorkerResult:
    job = find_next_queued_job(runs_directory)
    if job is None:
        return WorkerResult(processed=False)
    if not claim_job(job):
        return WorkerResult(processed=False)

    active_runner = runner or LocalJobRunner(background=False)
    if job.stage == "ingest":
        active_runner.submit_ingest(job.run_directory)
    else:
        active_runner.submit_stage(_stage_request(job))

    manifest = RunManifest.load(job.run_directory)
    return WorkerResult(
        processed=True,
        run_id=job.run_id,
        stage=job.stage,
        status=manifest.status.value,
    )


def run_worker_loop(
    *,
    runs_directory: Path,
    poll_seconds: float,
    runner: JobRunner | None = None,
) -> None:
    if poll_seconds <= 0:
        raise WorkerError("Poll seconds must be greater than zero.")
    while True:
        result = run_worker_once(
            runs_directory=runs_directory,
            runner=runner,
        )
        if not result.processed:
            time.sleep(poll_seconds)


def claim_job(job: QueuedJob) -> bool:
    manifest = RunManifest.load(job.run_directory)
    record = manifest.stages.get(job.stage)
    if record is None or record.status != StageStatus.QUEUED:
        return False
    record.status = StageStatus.RUNNING
    record.started_at = datetime.now(timezone.utc)
    record.completed_at = None
    record.error = None
    manifest.status = RunStatus.RUNNING
    manifest.save(job.run_directory)
    return True


def _stage_request(job: QueuedJob) -> StageRequest:
    input_directory = job.run_directory / "input"
    glossary_path = input_directory / "glossary.json"
    voice_reference_path = input_directory / "voice-reference.json"
    return StageRequest(
        run_directory=job.run_directory,
        stage=job.stage,
        glossary_path=(
            glossary_path if job.stage == "localize" and glossary_path.exists()
            else None
        ),
        voice_reference_path=(
            voice_reference_path
            if job.stage == "synthesize" and voice_reference_path.exists()
            else None
        ),
    )

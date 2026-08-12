from pathlib import Path

from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.runner import StageRequest
from dub_mvp.worker import (
    QueuedJob,
    claim_job,
    find_next_queued_job,
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
        manifest = RunManifest.load(request.run_directory)
        record = manifest.stages[request.stage]
        record.status = StageStatus.COMPLETED
        manifest.status = {
            "transcribe": RunStatus.TRANSCRIBED,
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
        manifest.status = RunStatus.QUEUED
        manifest.stages[queued_stage].status = StageStatus.QUEUED
    manifest.save(run_directory)
    return manifest


def test_find_next_queued_job_uses_stage_order(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    manifest = write_manifest(run_directory, run_id="run-a")
    manifest.status = RunStatus.QUEUED
    manifest.stages["render"].status = StageStatus.QUEUED
    manifest.stages["transcribe"].status = StageStatus.QUEUED
    manifest.save(run_directory)

    job = find_next_queued_job(tmp_path)

    assert job == QueuedJob(
        run_directory=run_directory,
        run_id="run-a",
        stage="transcribe",
    )


def test_claim_job_marks_stage_running(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")

    assert claim_job(
        QueuedJob(run_directory=run_directory, run_id="run-a", stage="ingest")
    )
    manifest = RunManifest.load(run_directory)

    assert manifest.status == RunStatus.RUNNING
    assert manifest.stages["ingest"].status == StageStatus.RUNNING
    assert manifest.stages["ingest"].started_at is not None


def test_worker_once_processes_queued_ingest(tmp_path: Path) -> None:
    run_directory = tmp_path / "run-a"
    write_manifest(run_directory, run_id="run-a", queued_stage="ingest")
    runner = RecordingRunner()

    result = run_worker_once(runs_directory=tmp_path, runner=runner)

    assert result.processed
    assert result.run_id == "run-a"
    assert result.stage == "ingest"
    assert result.status == "ingested"
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

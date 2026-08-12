from pathlib import Path

from dub_mvp.manifest import (
    RunManifest,
    RunStatus,
    StageStatus,
    mutate_manifest,
)
from dub_mvp.runner import LocalJobRunner, StageRequest
from dub_mvp.transcribe import TranscriptionError


class ConcurrentlyWrittenPipeline:
    """Fail a stage after something else has already written the manifest.

    This is what a heartbeat or a cancellation looks like from the runner's
    point of view: the manifest it started with is no longer current.
    """

    def run(self, *, run_directory: Path, **_):
        def touch(manifest: RunManifest) -> None:
            manifest.stages["transcribe"].heartbeat_at = manifest.updated_at

        mutate_manifest(run_directory, touch)
        raise TranscriptionError("model unavailable")


class ConcurrentlyWrittenSuccessPipeline:
    def run(self, *, run_directory: Path, **_):
        def touch(manifest: RunManifest) -> None:
            manifest.stages["transcribe"].heartbeat_at = manifest.updated_at

        mutate_manifest(run_directory, touch)
        transcript = run_directory / "metadata" / "transcript.json"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text("{}", encoding="utf-8")

        class _Transcript:
            model = "fake-whisperx"

        return _Transcript(), [], {"transcript": str(transcript)}


def prepared_run(tmp_path: Path) -> Path:
    run_directory = tmp_path / "run-a"
    audio = run_directory / "working" / "source_audio.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    manifest = RunManifest(
        run_id="run-a",
        source_path=str(run_directory / "input.mp4"),
        source_start_ms=0,
        source_end_ms=90000,
    )
    manifest.outputs["working_audio"] = str(audio)
    manifest.save(run_directory)
    return run_directory


def test_stage_failure_is_recorded_when_the_manifest_changed_mid_stage(
    tmp_path: Path,
) -> None:
    run_directory = prepared_run(tmp_path)
    runner = LocalJobRunner(
        transcription_pipeline=ConcurrentlyWrittenPipeline(),
        background=False,
    )

    runner.submit_stage(
        StageRequest(run_directory=run_directory, stage="transcribe")
    )

    manifest = RunManifest.load(run_directory)
    record = manifest.stages["transcribe"]
    assert record.status == StageStatus.FAILED
    assert record.error == "model unavailable"
    assert record.error_class == "TranscriptionError"
    assert manifest.status == RunStatus.FAILED
    assert manifest.errors == ["model unavailable"]


def test_unexpected_error_fails_the_stage_terminally(tmp_path: Path) -> None:
    class BrokenPipeline:
        def run(self, **_):
            raise KeyError("speaker_id")

    run_directory = prepared_run(tmp_path)
    runner = LocalJobRunner(
        transcription_pipeline=BrokenPipeline(),
        background=False,
    )

    runner.submit_stage(
        StageRequest(run_directory=run_directory, stage="transcribe")
    )

    record = RunManifest.load(run_directory).stages["transcribe"]
    assert record.status == StageStatus.FAILED
    assert record.error_class == "unexpected_error"
    assert "KeyError" in (record.error or "")
    assert not record.retryable


def test_stage_success_is_recorded_when_the_manifest_changed_mid_stage(
    tmp_path: Path,
) -> None:
    run_directory = prepared_run(tmp_path)
    runner = LocalJobRunner(
        transcription_pipeline=ConcurrentlyWrittenSuccessPipeline(),
        background=False,
    )

    runner.submit_stage(
        StageRequest(run_directory=run_directory, stage="transcribe")
    )

    manifest = RunManifest.load(run_directory)
    record = manifest.stages["transcribe"]
    assert record.status == StageStatus.COMPLETED
    assert manifest.status == RunStatus.TRANSCRIBED
    assert manifest.models["whisperx"] == "fake-whisperx"
    assert record.duration_seconds is not None

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import MediaMetadata, RunManifest, RunStatus, StageStatus
from dub_mvp.runner import QueuedJobRunner
from dub_mvp.webapp import (
    WebJobService,
    _build_runner,
    _parse_multipart,
    _safe_filename,
)


class FakeIngestor:
    def ingest(
        self,
        *,
        source: Path,
        run_directory: Path,
        start_ms: int,
        end_ms: int,
    ):
        assert source.is_file()
        assert start_ms == 0
        assert end_ms == 90000
        working = run_directory / "working"
        metadata = run_directory / "metadata"
        working.mkdir(parents=True)
        metadata.mkdir(parents=True)
        source_segment = working / "source_segment.mp4"
        working_audio = working / "source_audio.wav"
        probe = metadata / "ffprobe.json"
        source_segment.write_bytes(b"video")
        working_audio.write_bytes(b"audio")
        probe.write_text("{}\n", encoding="utf-8")
        return MediaMetadata(
            duration_seconds=120,
            video_codec="h264",
            width=1920,
            height=1080,
            audio_codec="aac",
        ), {
            "probe": str(probe),
            "source_segment": str(source_segment),
            "working_audio": str(working_audio),
        }


class FakeTranscriptionPipeline:
    def run(self, *, run_directory: Path, **_):
        transcript = run_directory / "metadata" / "transcript.json"
        segments = run_directory / "metadata" / "segments.json"
        raw = run_directory / "metadata" / "raw.json"
        transcript.write_text("{}", encoding="utf-8")
        segments.write_text("[]", encoding="utf-8")
        raw.write_text("{}", encoding="utf-8")
        return SimpleNamespace(model="fake-whisperx"), [], {
            "transcript": str(transcript),
            "segments": str(segments),
            "whisperx_raw": str(raw),
        }


class FakeLocalizationPipeline:
    def run(self, *, run_directory: Path, **_):
        raw = run_directory / "metadata" / "localization_raw.json"
        localized = run_directory / "metadata" / "localized_segments.json"
        raw.write_text("{}", encoding="utf-8")
        localized.write_text("[]", encoding="utf-8")
        return [], {
            "localization_raw": str(raw),
            "localized_segments": str(localized),
        }, "fake-translator"


class FakeSynthesisPipeline:
    def run(self, *, run_directory: Path, **_):
        raw = run_directory / "metadata" / "synthesis_raw.json"
        synthesized = run_directory / "metadata" / "synthesized_segments.json"
        raw.write_text("{}", encoding="utf-8")
        synthesized.write_text("[]", encoding="utf-8")
        return [], {
            "synthesis_raw": str(raw),
            "synthesized_segments": str(synthesized),
        }, "fake-tts"


class FakeRenderPipeline:
    def run(self, *, run_directory: Path, **_):
        plan = run_directory / "metadata" / "alignment_plan.json"
        srt = run_directory / "subtitles" / "hi.srt"
        audio = run_directory / "working" / "dubbed_audio.wav"
        video = run_directory / "outputs" / "dubbed_video.mp4"
        for path in (srt, audio, video):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"out")
        plan.write_text('{"segments":[]}', encoding="utf-8")
        return SimpleNamespace(segments=[]), {
            "alignment_plan": str(plan),
            "hindi_srt": str(srt),
            "dubbed_audio": str(audio),
            "dubbed_video": str(video),
        }


def multipart_body() -> tuple[bytes, str]:
    boundary = "test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="language"\r\n'
        "\r\n"
        "hi\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="video"; filename="demo video.mp4"\r\n'
        "Content-Type: video/mp4\r\n"
        "\r\n"
        "video-bytes\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def test_parse_multipart_upload() -> None:
    body, content_type = multipart_body()

    parts = _parse_multipart(body, content_type)

    assert parts["language"].content == b"hi"
    assert parts["video"].filename == "demo video.mp4"
    assert parts["video"].content == b"video-bytes"


def test_web_job_service_creates_run_and_ingests(tmp_path: Path) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        start_background_jobs=False,
    )

    payload = service.create_job(
        filename="demo video.mp4",
        content=b"video-bytes",
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    manifest = RunManifest.load(run_directory)

    assert manifest.status == RunStatus.INGESTED
    assert manifest.target_language == "hi"
    assert manifest.stages["ingest"].status == StageStatus.COMPLETED
    assert Path(manifest.outputs["working_audio"]).is_file()


def test_web_job_service_runs_full_customer_lifecycle(tmp_path: Path) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        transcription_pipeline=FakeTranscriptionPipeline(),
        localization_pipeline=FakeLocalizationPipeline(),
        synthesis_pipeline=FakeSynthesisPipeline(),
        render_pipeline=FakeRenderPipeline(),
        start_background_jobs=False,
    )
    payload = service.create_job(
        filename="demo.mp4",
        content=b"video",
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id

    service.run_stage(run_id=run_id, stage="transcribe")
    assert RunManifest.load(run_directory).status == RunStatus.TRANSCRIBED
    service.run_stage(run_id=run_id, stage="localize")
    assert RunManifest.load(run_directory).status == RunStatus.LOCALIZED
    service.run_stage(run_id=run_id, stage="synthesize")
    assert RunManifest.load(run_directory).status == RunStatus.SYNTHESIZED
    final_payload = service.run_stage(run_id=run_id, stage="render")
    manifest = RunManifest.load(run_directory)

    assert manifest.status == RunStatus.RENDERED
    assert manifest.stages["render"].status == StageStatus.COMPLETED
    assert Path(manifest.outputs["dubbed_video"]).is_file()
    assert final_payload["summary"]["status"] == "rendered"
    assert final_payload["summary"]["outputs"]["dubbed_video"]


def test_web_job_service_can_queue_without_local_execution(tmp_path: Path) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        runner=QueuedJobRunner(),
    )

    payload = service.create_job(
        filename="demo.mp4",
        content=b"video",
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    manifest = RunManifest.load(run_directory)

    assert payload["summary"]["status"] == "queued"
    assert manifest.status == RunStatus.QUEUED
    assert manifest.stages["ingest"].status == StageStatus.QUEUED
    assert (run_directory / "metadata" / "job-queue.jsonl").is_file()


def test_web_job_service_queues_heavy_stage_for_remote_worker(
    tmp_path: Path,
) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        start_background_jobs=False,
    )
    payload = service.create_job(
        filename="demo.mp4",
        content=b"video",
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    service.runner = QueuedJobRunner()

    queued = service.run_stage(run_id=run_id, stage="transcribe")
    manifest = RunManifest.load(run_directory)

    assert queued["summary"]["status"] == "queued"
    assert manifest.stages["transcribe"].status == StageStatus.QUEUED
    assert "transcribe" in (
        run_directory / "metadata" / "job-queue.jsonl"
    ).read_text(encoding="utf-8")


def test_build_runner_supports_remote_alias() -> None:
    assert isinstance(_build_runner("remote"), QueuedJobRunner)


def test_safe_filename_removes_path_and_spaces() -> None:
    assert _safe_filename("../demo video.mp4") == "demo-video.mp4"


def test_web_command_help_is_available() -> None:
    result = CliRunner().invoke(app, ["web", "--help"])

    assert result.exit_code == 0
    assert "customer video translation web app" in result.output

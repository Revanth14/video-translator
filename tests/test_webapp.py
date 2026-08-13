import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.artifacts import ArtifactMetadata, verify_artifact_integrity
from dub_mvp.configuration import PipelineConfigurationSnapshot
from dub_mvp.manifest import MediaMetadata, RunManifest, RunStatus, StageStatus
from dub_mvp.runner import QueuedJobRunner
from dub_mvp.upload import parse_multipart_stream
from dub_mvp.webapp import (
    WebJobService,
    WebAppError,
    _build_runner,
    _safe_filename,
)


class FakeIngestor:
    def inspect(self, source: Path) -> MediaMetadata:
        assert source.is_file()
        return MediaMetadata(
            duration_seconds=120,
            video_codec="h264",
            width=1920,
            height=1080,
            audio_codec="aac",
        )

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
        assert end_ms == 120000
        working = run_directory / "working"
        metadata = run_directory / "metadata"
        working.mkdir(parents=True, exist_ok=True)
        metadata.mkdir(parents=True, exist_ok=True)
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


class FakeUtterancePipeline:
    def run(self, *, run_directory: Path, **_):
        utterance_directory = run_directory / "utterances"
        utterance_directory.mkdir()
        artifact = utterance_directory / "dubbing_utterances.json"
        translation = utterance_directory / "translation_segments.json"
        sidecar = utterance_directory / "dubbing_utterances.meta.json"
        artifact.write_text('{"utterances":[]}', encoding="utf-8")
        translation.write_text("[]", encoding="utf-8")
        sidecar.write_text("{}", encoding="utf-8")
        return SimpleNamespace(utterances=[]), [], {
            "dubbing_utterances": str(artifact),
            "translation_segments": str(translation),
            "dubbing_utterances_metadata": str(sidecar),
        }


class FakeLocalizationPipeline:
    def run(self, *, run_directory: Path, segments_path: Path, **_):
        assert segments_path.name == "translation_segments.json"
        assert segments_path.is_file()
        raw = run_directory / "metadata" / "localization_raw.json"
        localized = run_directory / "metadata" / "localized_segments.json"
        metrics = run_directory / "metadata" / "translation_metrics.json"
        raw.write_text("{}", encoding="utf-8")
        localized.write_text("[]", encoding="utf-8")
        metrics.write_text(
            json.dumps(
                {
                    "provider": "fixture",
                    "model": "fake-translator",
                    "prompt_version": "semantic_translation_v1",
                    "configuration_fingerprint": "a" * 64,
                    "batch_count": 1,
                    "provider_calls": 1,
                    "reused_batches": 0,
                    "regenerated_batches": 0,
                    "attempt_count": 1,
                    "failed_attempts": 0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "provider_latency_seconds": 0.01,
                    "cost_usd": 0.001,
                    "cost_status": "reported",
                }
            ),
            encoding="utf-8",
        )
        return [], {
            "localization_raw": str(raw),
            "localized_segments": str(localized),
            "translation_metrics": str(metrics),
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


def staged_upload(tmp_path: Path, data: bytes = b"video") -> Path:
    """An upload already written to disk, as the streaming handler leaves it."""
    staged = tmp_path / ".uploads" / f"{uuid4().hex}.upload"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(data)
    return staged


def test_parse_multipart_upload() -> None:
    body, content_type = multipart_body()

    parts = parse_multipart_stream(
        BytesIO(body),
        content_type=content_type,
        content_length=len(body),
    )

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
        source_file=staged_upload(tmp_path, b"video-bytes"),
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    manifest = RunManifest.load(run_directory)

    assert manifest.status == RunStatus.QUEUED
    assert manifest.target_language == "hi"
    assert manifest.source_end_ms == 120000
    assert manifest.stages["ingest"].status == StageStatus.COMPLETED
    assert manifest.stages["transcribe"].status == StageStatus.QUEUED
    assert Path(manifest.outputs["working_audio"]).is_file()
    configuration_path = Path(manifest.outputs["configuration_snapshot"])
    configuration = PipelineConfigurationSnapshot.model_validate_json(
        configuration_path.read_text(encoding="utf-8")
    )
    assert configuration.source_language == "en"
    assert configuration.target_language == "hi"
    assert configuration.voice_catalog_sha256 is not None
    configuration_metadata = ArtifactMetadata.model_validate_json(
        Path(manifest.outputs["configuration_snapshot_metadata"]).read_text(
            encoding="utf-8"
        )
    )
    assert verify_artifact_integrity(
        configuration_metadata, root=run_directory
    ).valid


def test_web_job_rejects_language_without_passing_expansion_gate(
    tmp_path: Path,
) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        start_background_jobs=False,
    )
    upload = staged_upload(tmp_path, b"video-bytes")

    with pytest.raises(WebAppError, match="not release-enabled"):
        service.create_job(
            filename="demo.mp4",
            source_file=upload,
            target_language="ta",
        )

    assert upload.is_file()
    assert not list(path for path in tmp_path.iterdir() if path.name != ".uploads")


def test_web_job_service_runs_full_customer_lifecycle(tmp_path: Path) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        transcription_pipeline=FakeTranscriptionPipeline(),
        utterance_pipeline=FakeUtterancePipeline(),
        localization_pipeline=FakeLocalizationPipeline(),
        synthesis_pipeline=FakeSynthesisPipeline(),
        render_pipeline=FakeRenderPipeline(),
        start_background_jobs=False,
    )
    payload = service.create_job(
        filename="demo.mp4",
        source_file=staged_upload(tmp_path),
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    final_payload = payload
    manifest = RunManifest.load(run_directory)

    assert manifest.status == RunStatus.RENDERED
    assert manifest.stages["render"].status == StageStatus.COMPLETED
    localize = manifest.stages["localize"]
    assert localize.provider == "fixture"
    assert localize.input_fingerprint == "a" * 64
    assert localize.cost_usd == 0.001
    assert Path(manifest.outputs["dubbed_video"]).is_file()
    assert final_payload["summary"]["status"] == "rendered"
    assert final_payload["summary"]["outputs"]["dubbed_video"]


def test_web_job_service_persists_inputs_at_creation(tmp_path: Path) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        runner=QueuedJobRunner(),
    )

    payload = service.create_job(
        filename="demo.mp4",
        source_file=staged_upload(tmp_path),
        target_language="hi",
        glossary_content=b'{"terms":[]}',
        voice_reference_content=(
            b'{"reference_id":"voice-a","path":null,'
            b'"consent":"approved stock voice"}'
        ),
    )
    input_directory = tmp_path / payload["summary"]["run_id"] / "input"

    assert json.loads(
        (input_directory / "glossary.json").read_text(encoding="utf-8")
    ) == {"terms": []}
    assert json.loads(
        (input_directory / "translation-context.json").read_text(
            encoding="utf-8"
        )
    )["tone"] == "natural conversational speech"
    assert b'"voice-a"' in (
        input_directory / "voice-reference.json"
    ).read_bytes()


def test_web_job_service_rejects_invalid_voice_catalog_before_admission(
    tmp_path: Path,
) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        runner=QueuedJobRunner(),
    )

    with pytest.raises(WebAppError, match="voice reference"):
        service.create_job(
            filename="demo.mp4",
            source_file=staged_upload(tmp_path),
            target_language="hi",
            voice_reference_content=b'{"reference_id":"voice-a"}',
        )

    assert not list(tmp_path.glob("web-*"))


def test_web_job_service_rejects_invalid_translation_context_before_admission(
    tmp_path: Path,
) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        runner=QueuedJobRunner(),
    )

    with pytest.raises(WebAppError, match="translation context"):
        service.create_job(
            filename="demo.mp4",
            source_file=staged_upload(tmp_path),
            target_language="hi",
            translation_context_content=b'{"tone":""}',
        )

    assert not list(tmp_path.glob("*/manifest.json"))


def test_web_job_service_can_queue_without_local_execution(tmp_path: Path) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        runner=QueuedJobRunner(),
    )

    payload = service.create_job(
        filename="demo.mp4",
        source_file=staged_upload(tmp_path),
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    manifest = RunManifest.load(run_directory)

    assert payload["summary"]["status"] == "queued"
    assert manifest.status == RunStatus.QUEUED
    assert manifest.stages["ingest"].status == StageStatus.QUEUED
    assert (run_directory / "metadata" / "job-queue.jsonl").is_file()


def test_web_job_service_does_not_duplicate_already_queued_stage(
    tmp_path: Path,
) -> None:
    service = WebJobService(
        runs_directory=tmp_path,
        ingestor=FakeIngestor(),
        start_background_jobs=False,
    )
    payload = service.create_job(
        filename="demo.mp4",
        source_file=staged_upload(tmp_path),
        target_language="hi",
    )
    run_id = payload["summary"]["run_id"]
    run_directory = tmp_path / run_id
    service.runner = QueuedJobRunner()

    queued = service.run_stage(run_id=run_id, stage="transcribe")
    manifest = RunManifest.load(run_directory)

    assert queued["summary"]["status"] == "queued"
    assert manifest.stages["transcribe"].status == StageStatus.QUEUED
    events = (
        run_directory / "metadata" / "job-queue.jsonl"
    ).read_text(encoding="utf-8")
    assert events.count('"stage": "transcribe"') == 0


def test_build_runner_supports_remote_alias() -> None:
    assert isinstance(_build_runner("remote"), QueuedJobRunner)


def test_safe_filename_removes_path_and_spaces() -> None:
    assert _safe_filename("../demo video.mp4") == "demo-video.mp4"


def test_web_command_help_is_available() -> None:
    result = CliRunner().invoke(app, ["web", "--help"])

    assert result.exit_code == 0
    assert "customer video translation web app" in result.output

import json
from pathlib import Path

from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.transcribe import (
    TranscriptionPipeline,
    build_timestamped_segments,
    normalize_whisperx_result,
)


FIXTURE = Path(__file__).parent / "fixtures" / "whisperx_smoke.json"


class FixtureProvider:
    calls = 0

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or json.loads(FIXTURE.read_text(encoding="utf-8"))

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        duration_ms: int,
    ) -> dict:
        assert audio_path.is_file()
        assert language == "en"
        assert duration_ms == 10000
        self.calls += 1
        return self.payload


def test_normalizes_whisperx_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    transcript = normalize_whisperx_result(
        payload,
        duration_ms=10000,
        language="en",
        model_name="large-v3",
    )

    assert transcript.language == "en"
    assert transcript.model == "large-v3"
    assert [utterance.utterance_id for utterance in transcript.utterances] == [
        "utt_0001",
        "utt_0002",
    ]
    assert transcript.utterances[0].start_ms == 120
    assert transcript.utterances[0].end_ms == 3400
    assert transcript.utterances[0].words[3].text == "API"


def test_builds_deterministic_timestamped_segments() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    transcript = normalize_whisperx_result(
        payload,
        duration_ms=10000,
        language="en",
        model_name="large-v3",
    )

    first = build_timestamped_segments(transcript)
    second = build_timestamped_segments(transcript)

    assert [segment.model_dump() for segment in first] == [
        segment.model_dump() for segment in second
    ]
    assert [segment.segment_id for segment in first] == [
        "seg_0001",
        "seg_0002",
    ]
    assert [(segment.start_ms, segment.end_ms) for segment in first] == [
        (120, 3400),
        (4200, 8600),
    ]
    assert first[0].duration_budget_ms == 3280
    assert first[1].source_text == "First, we build the Docker image."


def test_pipeline_writes_raw_transcript_and_segments(tmp_path: Path) -> None:
    audio = tmp_path / "working.wav"
    audio.touch()
    provider = FixtureProvider()

    transcript, segments, outputs = TranscriptionPipeline(
        provider=provider,
    ).run(
        audio_path=audio,
        run_directory=tmp_path,
        language="en",
        duration_ms=10000,
    )

    assert provider.calls == 1
    assert transcript.utterances
    assert len(segments) == 2
    assert Path(outputs["whisperx_raw"]).is_file()
    assert Path(outputs["transcript"]).is_file()
    assert Path(outputs["segments"]).is_file()


def test_transcribe_command_resumes_completed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "working" / "source_audio.wav"
    audio.parent.mkdir()
    audio.touch()
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["working_audio"] = str(audio)
    manifest.stages["ingest"].status = StageStatus.COMPLETED
    manifest.save(tmp_path)

    provider = FixtureProvider()

    class FakePipeline:
        def __init__(self, *, model_name: str) -> None:
            assert model_name == "large-v3"

        def run(self, **kwargs):
            return TranscriptionPipeline(provider=provider).run(**kwargs)

    monkeypatch.setattr("dub_mvp.cli.TranscriptionPipeline", FakePipeline)
    runner = CliRunner()

    first = runner.invoke(app, ["transcribe", str(tmp_path)])
    second = runner.invoke(app, ["transcribe", str(tmp_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already complete" in second.output
    assert provider.calls == 1

    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.TRANSCRIBED
    assert loaded.stages["transcribe"].status == StageStatus.COMPLETED
    assert Path(loaded.outputs["segments"]).is_file()

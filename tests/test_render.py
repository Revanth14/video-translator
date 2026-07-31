import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.render import (
    RenderError,
    RenderPipeline,
    build_render_plan,
    build_srt,
    load_synthesized_segments,
)


FIXTURES = Path(__file__).parent / "fixtures"
SYNTHESIZED = FIXTURES / "synthesized_segments_smoke.json"


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(
        self,
        command: list[str],
        **_: Any,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_bytes(b"render output")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def resolver(name: str) -> str:
    return f"/fake/{name}"


def write_synthesized_fixture(tmp_path: Path) -> Path:
    first_audio = tmp_path / "segments" / "seg_0001" / "tts-r1.wav"
    second_audio = tmp_path / "segments" / "seg_0002" / "tts-r1.wav"
    first_audio.parent.mkdir(parents=True)
    second_audio.parent.mkdir(parents=True)
    first_audio.write_bytes(b"audio 1")
    second_audio.write_bytes(b"audio 2")

    payload = json.loads(SYNTHESIZED.read_text(encoding="utf-8"))
    payload[0]["tts_audio_path"] = str(first_audio)
    payload[1]["tts_audio_path"] = str(second_audio)
    path = tmp_path / "metadata" / "synthesized_segments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builds_render_plan_with_review_threshold(tmp_path: Path) -> None:
    path = write_synthesized_fixture(tmp_path)
    segments = load_synthesized_segments(path)

    plan = build_render_plan(segments, duration_ms=10000)

    assert [segment.segment_id for segment in plan.segments] == [
        "seg_0001",
        "seg_0002",
    ]
    assert plan.segments[0].tempo_ratio == 0.969512
    assert plan.segments[0].needs_review is False
    assert plan.segments[1].tempo_ratio == 1.1
    assert plan.segments[1].needs_review is False


def test_build_render_plan_rejects_hard_tempo_limit(tmp_path: Path) -> None:
    path = write_synthesized_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[0]["tts_duration_ms"] = 4101
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RenderError, match="20% hard limit"):
        build_render_plan(load_synthesized_segments(path), duration_ms=10000)


def test_build_srt_uses_target_text_and_source_timing(tmp_path: Path) -> None:
    segments = load_synthesized_segments(write_synthesized_fixture(tmp_path))

    assert build_srt(segments) == (
        "1\n"
        "00:00:00,120 --> 00:00:03,400\n"
        "API deployment demo mein swagat hai.\n"
        "\n"
        "2\n"
        "00:00:04,200 --> 00:00:08,600\n"
        "Pehle, hum Docker image build karte hain.\n"
    )


def test_render_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    synthesized_path = write_synthesized_fixture(tmp_path)
    source_segment = tmp_path / "working" / "source_segment.mp4"
    source_segment.parent.mkdir()
    source_segment.write_bytes(b"video")
    runner = FakeRunner()

    plan, outputs = RenderPipeline(
        runner=runner,
        resolver=resolver,
    ).run(
        synthesized_segments_path=synthesized_path,
        source_segment_path=source_segment,
        run_directory=tmp_path,
        duration_ms=10000,
    )

    assert len(plan.segments) == 2
    assert Path(outputs["alignment_plan"]).is_file()
    assert Path(outputs["hindi_srt"]).is_file()
    assert Path(outputs["dubbed_audio"]).is_file()
    assert Path(outputs["dubbed_video"]).is_file()
    assert len(runner.commands) == 2
    assert "-filter_complex" in runner.commands[0]
    assert "adelay=120|120" in " ".join(runner.commands[0])


def test_render_command_resumes_completed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    synthesized_path = write_synthesized_fixture(tmp_path)
    source_segment = tmp_path / "working" / "source_segment.mp4"
    source_segment.parent.mkdir(exist_ok=True)
    source_segment.write_bytes(b"video")
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["synthesized_segments"] = str(synthesized_path)
    manifest.outputs["source_segment"] = str(source_segment)
    manifest.stages["synthesize"].status = StageStatus.COMPLETED
    manifest.save(tmp_path)
    runner = FakeRunner()

    class FakePipeline:
        def run(self, **kwargs):
            return RenderPipeline(runner=runner, resolver=resolver).run(**kwargs)

    monkeypatch.setattr("dub_mvp.cli.RenderPipeline", FakePipeline)
    cli_runner = CliRunner()

    first = cli_runner.invoke(app, ["render", str(tmp_path)])
    second = cli_runner.invoke(app, ["render", str(tmp_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already complete" in second.output
    assert len(runner.commands) == 2

    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.RENDERED
    assert loaded.stages["render"].status == StageStatus.COMPLETED
    assert Path(loaded.outputs["dubbed_video"]).is_file()


def test_render_command_failure_updates_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    synthesized_path = write_synthesized_fixture(tmp_path)
    source_segment = tmp_path / "working" / "source_segment.mp4"
    source_segment.parent.mkdir(exist_ok=True)
    source_segment.write_bytes(b"video")
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["synthesized_segments"] = str(synthesized_path)
    manifest.outputs["source_segment"] = str(source_segment)
    manifest.save(tmp_path)

    class FakePipeline:
        def run(self, **kwargs):
            raise RenderError("render exploded")

    monkeypatch.setattr("dub_mvp.cli.RenderPipeline", FakePipeline)
    result = CliRunner().invoke(app, ["render", str(tmp_path)])

    assert result.exit_code == 1
    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.FAILED
    assert loaded.stages["render"].status == StageStatus.FAILED
    assert loaded.stages["render"].error == "render exploded"

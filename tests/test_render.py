import json
import math
import os
import shutil
import subprocess
import wave
from multiprocessing import Process
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.render import (
    RenderError,
    RenderPipeline,
    RenderPolicy,
    RenderReport,
    CompositionMode,
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
        if command[0].endswith("ffprobe"):
            path = Path(command[-1])
            if path.suffix == ".wav":
                streams = [
                    {
                        "codec_type": "audio",
                        "codec_name": "pcm_s16le",
                        "sample_rate": "48000",
                        "channels": 2,
                        "duration": "10.0",
                    }
                ]
                format_name = "wav"
            else:
                streams = [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30/1",
                        "duration": "10.0",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                        "duration": "10.0",
                    },
                ]
                format_name = "mov,mp4,m4a,3gp,3g2,mj2"
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "format": {
                            "duration": "10.0",
                            "format_name": format_name,
                        },
                        "streams": streams,
                    }
                ),
                stderr="",
            )
        if command[-1] != "-":
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_bytes(b"render output")
        stderr = "max_volume: -1.0 dB" if "volumedetect" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=stderr)


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
    assert len(runner.commands) == 8
    assembly = next(command for command in runner.commands if "-filter_complex" in command)
    assert "anullsrc" in " ".join(assembly)
    assert "loudnorm" in " ".join(assembly)
    assert "adelay=120:all=1" in " ".join(assembly)
    mux = next(command for command in runner.commands if "+faststart" in command)
    assert "-shortest" not in mux
    assert Path(outputs["render_report"]).is_file()
    assert Path(outputs["dubbed_video_metadata"]).is_file()


def test_render_pipeline_reuses_verified_outputs_without_new_commands(
    tmp_path: Path,
) -> None:
    synthesized_path = write_synthesized_fixture(tmp_path)
    source_segment = tmp_path / "working" / "source_segment.mp4"
    source_segment.parent.mkdir()
    source_segment.write_bytes(b"video")
    runner = FakeRunner()
    pipeline = RenderPipeline(runner=runner, resolver=resolver)

    _, first = pipeline.run(
        synthesized_segments_path=synthesized_path,
        source_segment_path=source_segment,
        run_directory=tmp_path,
        duration_ms=10000,
    )
    _, second = pipeline.run(
        synthesized_segments_path=synthesized_path,
        source_segment_path=source_segment,
        run_directory=tmp_path,
        duration_ms=10000,
    )

    assert len(runner.commands) == 8
    assert first == second


def test_corrupt_video_gets_new_revision_without_rebuilding_valid_audio(
    tmp_path: Path,
) -> None:
    synthesized_path = write_synthesized_fixture(tmp_path)
    source_segment = tmp_path / "working" / "source_segment.mp4"
    source_segment.parent.mkdir()
    source_segment.write_bytes(b"video")
    runner = FakeRunner()
    pipeline = RenderPipeline(runner=runner, resolver=resolver)
    _, first = pipeline.run(
        synthesized_segments_path=synthesized_path,
        source_segment_path=source_segment,
        run_directory=tmp_path,
        duration_ms=10000,
    )
    Path(first["dubbed_video"]).write_bytes(b"corrupt")
    before = len(runner.commands)

    _, second = pipeline.run(
        synthesized_segments_path=synthesized_path,
        source_segment_path=source_segment,
        run_directory=tmp_path,
        duration_ms=10000,
    )

    new_commands = runner.commands[before:]
    assert first["dubbed_audio"] == second["dubbed_audio"]
    assert first["dubbed_video"] != second["dubbed_video"]
    assert not any("-filter_complex" in command for command in new_commands)
    assert any("+faststart" in command for command in new_commands)


def test_original_track_ducking_is_explicit_and_windowed(tmp_path: Path) -> None:
    synthesized_path = write_synthesized_fixture(tmp_path)
    source_segment = tmp_path / "working" / "source_segment.mp4"
    source_segment.parent.mkdir()
    source_segment.write_bytes(b"video")
    runner = FakeRunner()
    policy = RenderPolicy(composition_mode=CompositionMode.DUCK_ORIGINAL)

    _, outputs = RenderPipeline(
        runner=runner,
        resolver=resolver,
        policy=policy,
    ).run(
        synthesized_segments_path=synthesized_path,
        source_segment_path=source_segment,
        run_directory=tmp_path,
        duration_ms=10000,
    )

    assembly = next(command for command in runner.commands if "-filter_complex" in command)
    filter_graph = assembly[assembly.index("-filter_complex") + 1]
    assert str(source_segment) in assembly
    assert "volume='if(gt(between(t,0.120,3.400)" in filter_graph
    assert f",{policy.duck_volume:.6f},1.0)'" in filter_graph
    report = RenderReport.model_validate_json(
        Path(outputs["render_report"]).read_text(encoding="utf-8")
    )
    assert report.composition_mode == CompositionMode.DUCK_ORIGINAL


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
    assert len(runner.commands) == 8

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


def _write_tone_wav(path: Path, duration_ms: int) -> None:
    frame_rate = 48_000
    frames = bytearray()
    for index in range(duration_ms * frame_rate // 1000):
        sample = round(2500 * math.sin(2 * math.pi * 330 * index / frame_rate))
        frames.extend(int(sample).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(bytes(frames))


def _real_render_fixture(tmp_path: Path) -> tuple[Path, Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")
    source = tmp_path / "working" / "source_segment.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=24:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr
    speech = tmp_path / "speech" / "tts.wav"
    speech.parent.mkdir()
    _write_tone_wav(speech, 800)
    synthesized = tmp_path / "speech" / "synthesized.json"
    synthesized.write_text(
        json.dumps(
            [
                {
                    "schema_version": 3,
                    "segment_id": "utt_0001",
                    "start_ms": 500,
                    "end_ms": 1300,
                    "duration_budget_ms": 800,
                    "speaker_id": "speaker_01",
                    "source_text": "source",
                    "target_text": "लक्ष्य",
                    "target_text_revision": 1,
                    "tts_audio_path": str(speech),
                    "tts_duration_ms": 800,
                    "model": "fixture",
                    "reference_id": "voice_A",
                    "original_tts_audio_path": str(speech),
                    "original_tts_duration_ms": 800,
                    "duration_error_ms": 0,
                    "duration_ratio": 1.0,
                    "duration_status": "accepted",
                    "duration_strategy": "accept",
                    "duration_correction_path": "speech/duration/result.json",
                    "duration_correction_metadata_path": "speech/duration/result.meta.json",
                }
            ]
        ),
        encoding="utf-8",
    )
    return source, synthesized


def test_real_ffmpeg_render_is_full_duration_decodable_and_preserves_video(
    tmp_path: Path,
) -> None:
    source, synthesized = _real_render_fixture(tmp_path)

    plan, outputs = RenderPipeline().run(
        synthesized_segments_path=synthesized,
        source_segment_path=source,
        run_directory=tmp_path,
        duration_ms=3000,
    )

    report = RenderReport.model_validate_json(
        Path(outputs["render_report"]).read_text(encoding="utf-8")
    )
    assert plan.duration_ms == 3000
    assert report.validation.passed
    assert report.validation.full_decode_succeeded
    assert report.validation.duration_within_tolerance
    assert report.validation.video_stream_copied
    assert report.validation.output_width == 160
    assert report.validation.output_height == 90
    assert report.validation.audio_sample_rate_hz == 48_000
    assert report.validation.audio_channels == 2
    assert not report.validation.clipping_detected
    assert "-shortest" not in json.dumps(
        json.loads(Path(outputs["render_commands"]).read_text(encoding="utf-8"))
    )


class _KillBeforeMuxRunner:
    def __call__(self, command: list[str], **kwargs: Any):
        if "+faststart" in command:
            os._exit(37)
        return subprocess.run(command, **kwargs)


def _render_until_process_death(
    run_directory: str, source: str, synthesized: str
) -> None:
    RenderPipeline(runner=_KillBeforeMuxRunner()).run(
        synthesized_segments_path=Path(synthesized),
        source_segment_path=Path(source),
        run_directory=Path(run_directory),
        duration_ms=3000,
    )


def test_render_resumes_verified_audio_after_real_process_death(
    tmp_path: Path,
) -> None:
    source, synthesized = _real_render_fixture(tmp_path)
    process = Process(
        target=_render_until_process_death,
        args=(str(tmp_path), str(source), str(synthesized)),
    )
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 37
    completed_audio = list(tmp_path.glob("working/dubbed-audio-*.wav"))
    assert len(completed_audio) == 1
    checksum_before = completed_audio[0].read_bytes()

    _, outputs = RenderPipeline().run(
        synthesized_segments_path=synthesized,
        source_segment_path=source,
        run_directory=tmp_path,
        duration_ms=3000,
    )

    assert Path(outputs["dubbed_audio"]) == completed_audio[0]
    assert completed_audio[0].read_bytes() == checksum_before
    attempts = json.loads(
        Path(outputs["render_commands"]).read_text(encoding="utf-8")
    )
    assert any(item.get("error_class") == "interrupted" for item in attempts)
    assert Path(outputs["dubbed_video"]).is_file()

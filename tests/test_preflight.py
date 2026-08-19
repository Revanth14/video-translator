from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import MediaMetadata, RunManifest
from dub_mvp.preflight import (
    PreflightCheck,
    PreflightProfile,
    build_preflight_report,
    report_to_json,
)


FIXTURES = Path(__file__).parent / "fixtures"
VOICE_REFERENCE = FIXTURES / "voice_reference_smoke.json"


def test_preflight_reports_missing_required_tools(monkeypatch) -> None:
    monkeypatch.setattr("dub_mvp.preflight.shutil.which", lambda _: None)

    report = build_preflight_report()

    assert report.ok is False
    assert report.checks[0].name == "tool:ffmpeg"
    assert report.checks[0].status == "fail"


def test_preflight_run_checks_warn_for_missing_future_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.save(tmp_path)

    report = build_preflight_report(run_directory=tmp_path)

    assert report.ok is True
    assert any(
        check.name == "run:output:working_audio"
        and check.status == "warn"
        for check in report.checks
    )


def test_preflight_validates_voice_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )

    report = build_preflight_report(voice_reference_path=VOICE_REFERENCE)

    assert report.ok is True
    assert any(
        check.name == "voice_reference" and check.status == "pass"
        for check in report.checks
    )


def test_preflight_command_exits_nonzero_for_blocking_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr("dub_mvp.preflight.shutil.which", lambda _: None)

    result = CliRunner().invoke(app, ["preflight"])

    assert result.exit_code == 1
    assert '"ok": false' in result.output


def test_benchmark_preflight_turns_missing_runtime_into_blockers(
    monkeypatch,
) -> None:
    monkeypatch.setattr("dub_mvp.preflight.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "dub_mvp.preflight.importlib.util.find_spec",
        lambda _: None,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(
        "VIDEO_TRANSLATOR_OPENAI_INPUT_USD_PER_MILLION",
        raising=False,
    )
    monkeypatch.delenv(
        "VIDEO_TRANSLATOR_OPENAI_OUTPUT_USD_PER_MILLION",
        raising=False,
    )

    report = build_preflight_report(profile=PreflightProfile.BENCHMARK)

    assert report.ok is False
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["python:whisperx"] == "fail"
    assert statuses["python:openai"] == "fail"
    assert statuses["python:indicf5"] == "fail"
    assert statuses["python:torch"] == "fail"
    assert statuses["env:OPENAI_API_KEY"] == "fail"
    assert statuses["tool:nvidia-smi"] == "fail"
    assert statuses["runtime:cuda"] == "fail"
    assert statuses["env:translation_pricing"] == "fail"
    assert statuses["benchmark:input"] == "fail"
    assert statuses["voice_reference"] == "fail"


def test_benchmark_preflight_accepts_complete_measured_setup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeIngestor:
        def inspect(self, path: Path) -> MediaMetadata:
            assert path == input_video
            return MediaMetadata(
                duration_seconds=35 * 60,
                format_name="mov,mp4",
                video_codec="h264",
                width=1920,
                height=1080,
                frame_rate="30000/1001",
                audio_codec="aac",
                audio_channels=2,
                audio_sample_rate=48000,
            )

    input_video = tmp_path / "authorized.mp4"
    input_video.write_bytes(b"fixture")
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )
    monkeypatch.setattr(
        "dub_mvp.preflight.importlib.util.find_spec",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "dub_mvp.preflight._cuda_check",
        lambda: PreflightCheck(
            name="runtime:cuda",
            status="pass",
            detail="fixture GPU",
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv(
        "VIDEO_TRANSLATOR_OPENAI_INPUT_USD_PER_MILLION",
        "0.25",
    )
    monkeypatch.setenv(
        "VIDEO_TRANSLATOR_OPENAI_OUTPUT_USD_PER_MILLION",
        "2.0",
    )

    report = build_preflight_report(
        profile=PreflightProfile.BENCHMARK,
        input_video_path=input_video,
        voice_reference_path=VOICE_REFERENCE,
        media_ingestor=FakeIngestor(),
    )

    assert report.ok is True
    assert report.profile == PreflightProfile.BENCHMARK
    assert all(check.status == "pass" for check in report.checks)


def test_benchmark_preflight_rejects_short_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class ShortIngestor:
        def inspect(self, _path: Path) -> MediaMetadata:
            return MediaMetadata(
                duration_seconds=5 * 60,
                video_codec="h264",
                width=1280,
                height=720,
                audio_codec="aac",
            )

    input_video = tmp_path / "short.mp4"
    input_video.write_bytes(b"fixture")
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )
    monkeypatch.setattr(
        "dub_mvp.preflight.importlib.util.find_spec",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "dub_mvp.preflight._cuda_check",
        lambda: PreflightCheck(
            name="runtime:cuda",
            status="pass",
            detail="fixture GPU",
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv(
        "VIDEO_TRANSLATOR_OPENAI_INPUT_USD_PER_MILLION",
        "0.25",
    )
    monkeypatch.setenv(
        "VIDEO_TRANSLATOR_OPENAI_OUTPUT_USD_PER_MILLION",
        "2.0",
    )

    report = build_preflight_report(
        profile=PreflightProfile.BENCHMARK,
        input_video_path=input_video,
        voice_reference_path=VOICE_REFERENCE,
        media_ingestor=ShortIngestor(),
    )

    input_check = next(
        check for check in report.checks if check.name == "benchmark:input"
    )
    assert report.ok is False
    assert input_check.status == "fail"
    assert "30-45 minutes" in input_check.detail


def test_report_to_json_outputs_newline() -> None:
    report = build_preflight_report()

    assert report_to_json(report).endswith("\n")

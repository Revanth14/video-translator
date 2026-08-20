import json
import wave
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
HINDI_REFERENCE_TEXT = (
    "मेरा नाम राहुल है और मैं इस वीडियो में आपको एक नई तकनीक के बारे में "
    "बताने जा रहा हूँ। यह बहुत आसान है।"
)


def write_reference_wav(path: Path, seconds: float, frame_rate: int = 24_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * frame_rate))


def write_voice_catalog(
    directory: Path,
    *,
    reference_seconds: float = 9.0,
    reference_text: str = HINDI_REFERENCE_TEXT,
) -> Path:
    audio_path = directory / "reference.wav"
    write_reference_wav(audio_path, reference_seconds)
    catalog_path = directory / "voice-catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "voices": [
                    {
                        "reference_id": "benchmark-reference",
                        "path": str(audio_path),
                        "reference_text": reference_text,
                        "consent": "approved fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


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


def test_preflight_warns_but_never_blocks_on_cross_script_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Source-clone dubbing prompts Hindi with English audio by design.

    Advisory in both profiles: blocking it would forbid the very configuration
    the product depends on, and quality there is decided by listening.
    """
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )
    monkeypatch.setattr(
        "dub_mvp.preflight._indicf5_runtime_check",
        lambda *, required: PreflightCheck(
            name="runtime:indicf5",
            status="pass",
            detail="fixture isolated TTS runtime",
        ),
    )
    catalog = write_voice_catalog(
        tmp_path,
        reference_text="My email is one at the rate gmail.com and it works.",
    )

    for profile in (PreflightProfile.LOCAL, PreflightProfile.BENCHMARK):
        report = build_preflight_report(
            profile=profile,
            voice_reference_path=catalog,
            target_language="hi",
        )
        check = next(
            item for item in report.checks if item.name == "voice_reference:prompt"
        )
        assert check.status == "warn", profile
        assert "devanagari" in check.detail


def test_benchmark_preflight_blocks_over_long_reference_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )
    catalog = write_voice_catalog(tmp_path, reference_seconds=15.0)

    report = build_preflight_report(
        voice_reference_path=catalog,
        target_language="hi",
    )
    local_check = next(
        item for item in report.checks if item.name == "voice_reference:prompt"
    )
    assert local_check.status == "warn"
    assert "clips anything over 12s" in local_check.detail

    monkeypatch.setattr(
        "dub_mvp.preflight._indicf5_runtime_check",
        lambda *, required: PreflightCheck(
            name="runtime:indicf5",
            status="pass",
            detail="fixture isolated TTS runtime",
        ),
    )
    benchmark_report = build_preflight_report(
        profile=PreflightProfile.BENCHMARK,
        voice_reference_path=catalog,
        target_language="hi",
    )

    benchmark_check = next(
        item
        for item in benchmark_report.checks
        if item.name == "voice_reference:prompt"
    )
    assert benchmark_report.ok is False
    assert benchmark_check.status == "fail"


def test_preflight_accepts_a_matched_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dub_mvp.preflight.shutil.which",
        lambda name: f"/fake/{name}",
    )

    report = build_preflight_report(
        voice_reference_path=write_voice_catalog(tmp_path),
        target_language="hi",
    )

    check = next(
        item for item in report.checks if item.name == "voice_reference:prompt"
    )
    assert check.status == "pass"
    assert "devanagari" in check.detail


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
    assert statuses["runtime:indicf5"] == "fail"
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
    monkeypatch.setattr(
        "dub_mvp.preflight._indicf5_runtime_check",
        lambda *, required: PreflightCheck(
            name="runtime:indicf5",
            status="pass",
            detail="fixture isolated TTS runtime",
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
        voice_reference_path=write_voice_catalog(tmp_path),
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

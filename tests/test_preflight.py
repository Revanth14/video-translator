from pathlib import Path

from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest
from dub_mvp.preflight import build_preflight_report, report_to_json


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


def test_report_to_json_outputs_newline() -> None:
    report = build_preflight_report()

    assert report_to_json(report).endswith("\n")

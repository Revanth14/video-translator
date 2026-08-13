import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.ui import (
    UiError,
    build_customer_run_payload,
    demo_payload,
    list_customer_runs,
    _safe_join,
)


def test_demo_payload_is_customer_ready() -> None:
    payload = demo_payload()

    assert payload["summary"]["status"] == "rendered"
    assert payload["summary"]["metrics"]["segments"] == 12
    assert payload["segments"][0]["target_text"]


def test_lists_customer_runs_newest_first(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    RunManifest(
        run_id="first",
        source_path="source-a.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    ).save(first)
    RunManifest(
        run_id="second",
        source_path="source-b.mp4",
        source_start_ms=0,
        source_end_ms=2000,
    ).save(second)

    runs = list_customer_runs(tmp_path)

    assert [run["run_id"] for run in runs] == ["second", "first"]
    assert runs[0]["duration_ms"] == 2000


def test_build_customer_run_payload_aggregates_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "demo-run"
    metadata = run / "metadata"
    subtitles = run / "subtitles"
    outputs = run / "outputs"
    for directory in (metadata, subtitles, outputs):
        directory.mkdir(parents=True)

    segments_path = metadata / "localized_segments.json"
    synthesized_path = metadata / "synthesized_segments.json"
    alignment_path = metadata / "alignment_plan.json"
    video_path = outputs / "dubbed_video.mp4"
    srt_path = subtitles / "hi.srt"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "seg_0001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "duration_budget_ms": 1000,
                    "source_text": "Hello.",
                    "target_text": "Namaste.",
                    "target_text_revision": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    synthesized_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "seg_0001",
                    "tts_duration_ms": 980,
                    "tts_audio_path": str(run / "segments/seg_0001/tts-r1.wav"),
                    "start_ms": 0,
                    "end_ms": 1000,
                    "duration_budget_ms": 1000,
                    "source_text": "Hello.",
                    "target_text": "Namaste.",
                    "target_text_revision": 1,
                    "tts_revision": 1,
                    "model": "fixture",
                    "reference_id": "voice",
                }
            ]
        ),
        encoding="utf-8",
    )
    alignment_path.write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "needs_review": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    video_path.write_bytes(b"video")
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nNamaste.\n")

    manifest = RunManifest(
        run_id="demo-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.status = RunStatus.RENDERED
    manifest.stages["render"].status = StageStatus.COMPLETED
    manifest.outputs.update(
        {
            "localized_segments": str(segments_path),
            "synthesized_segments": str(synthesized_path),
            "alignment_plan": str(alignment_path),
            "dubbed_video": str(video_path),
            "hindi_srt": str(srt_path),
        }
    )
    manifest.save(run)

    payload = build_customer_run_payload(run)

    assert payload["summary"]["metrics"]["localized"] == 1
    assert payload["summary"]["metrics"]["synthesized"] == 1
    assert payload["summary"]["metrics"]["needs_review"] == 1
    assert payload["segments"][0]["target_text"] == "Namaste."


def test_safe_join_rejects_paths_outside_runs(tmp_path: Path) -> None:
    with pytest.raises(UiError):
        _safe_join(tmp_path, "../outside.mp4")


def test_ui_command_help_includes_customer_review_language() -> None:
    result = CliRunner().invoke(app, ["ui", "--help"])

    assert result.exit_code == 0
    assert "customer-facing dubbing review UI" in result.output


def test_internal_ui_renders_durable_attempt_progress() -> None:
    from dub_mvp.ui import HTML

    assert "summary.stage_details" in HTML
    assert 'progress.attempts?.total' in HTML

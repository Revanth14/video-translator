import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.localize import (
    LocalizationError,
    LocalizationPipeline,
    load_glossary,
    load_transcript_segments,
    validate_localized_segments,
)
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus


FIXTURES = Path(__file__).parent / "fixtures"
SEGMENTS = FIXTURES / "segments_smoke.json"
GLOSSARY = FIXTURES / "glossary_smoke.json"


class FixtureTranslator:
    provider_name = "fixture"
    model_name = "fixture-hindi"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "target_text": "API deployment demo mein swagat hai.",
                    "terms_used": ["API"],
                },
                {
                    "segment_id": "seg_0002",
                    "target_text": "Pehle, hum Docker image build karte hain.",
                    "terms_used": ["Docker"],
                },
            ]
        }
        self.calls = 0
        self.last_glossary_terms: list[str] = []

    def localize(self, segments, *, source_language, target_language, glossary):
        assert [segment.segment_id for segment in segments] == [
            "seg_0001",
            "seg_0002",
        ]
        assert source_language == "en"
        assert target_language == "hi"
        self.calls += 1
        self.last_glossary_terms = [term.source for term in glossary.terms]
        return self.payload


def test_loads_glossary_fixture() -> None:
    glossary = load_glossary(GLOSSARY)

    assert [term.source for term in glossary.terms] == ["Docker", "API"]
    assert glossary.terms[0].target == "Docker"


def test_validates_localized_segments_in_source_order() -> None:
    source_segments = load_transcript_segments(SEGMENTS)
    localized = validate_localized_segments(
        source_segments,
        {
            "segments": [
                {
                    "segment_id": "seg_0002",
                    "target_text": "Pehle, hum Docker image build karte hain.",
                },
                {
                    "segment_id": "seg_0001",
                    "target_text": "API deployment demo mein swagat hai.",
                },
            ]
        },
    )

    assert [segment.segment_id for segment in localized] == [
        "seg_0001",
        "seg_0002",
    ]
    assert localized[0].source_text == "Welcome to the API deployment demo."
    assert localized[0].target_text == "API deployment demo mein swagat hai."
    assert localized[0].duration_budget_ms == 3280


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "segments": [
                    {"segment_id": "seg_0001", "target_text": "Hindi"},
                    {"segment_id": "seg_0001", "target_text": "Again"},
                ]
            },
            "Duplicate",
        ),
        (
            {
                "segments": [
                    {"segment_id": "seg_0001", "target_text": "Hindi"},
                ]
            },
            "Missing",
        ),
        (
            {
                "segments": [
                    {"segment_id": "seg_0001", "target_text": "Hindi"},
                    {"segment_id": "seg_9999", "target_text": "Unknown"},
                ]
            },
            "Unknown",
        ),
        (
            {
                "segments": [
                    {"segment_id": "seg_0001", "target_text": ""},
                    {"segment_id": "seg_0002", "target_text": "Hindi"},
                ]
            },
            "empty",
        ),
    ],
)
def test_rejects_invalid_localized_output(
    payload: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((LocalizationError, ValueError), match=message):
        validate_localized_segments(load_transcript_segments(SEGMENTS), payload)


def test_pipeline_writes_raw_and_localized_artifacts(tmp_path: Path) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    provider = FixtureTranslator()

    localized, outputs, model_name = LocalizationPipeline(
        provider=provider,
    ).run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )

    assert provider.calls == 1
    assert provider.last_glossary_terms == ["Docker", "API"]
    assert model_name == "fixture-hindi"
    assert len(localized) == 2
    assert Path(outputs["localization_raw"]).is_file()
    assert Path(outputs["localized_segments"]).is_file()


def test_localize_command_resumes_completed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    segments_path = tmp_path / "metadata" / "segments.json"
    segments_path.parent.mkdir()
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["segments"] = str(segments_path)
    manifest.stages["transcribe"].status = StageStatus.COMPLETED
    manifest.save(tmp_path)

    provider = FixtureTranslator()

    class FakePipeline:
        def __init__(self, *, model_name: str) -> None:
            assert model_name == "gpt-5-mini"

        def run(self, **kwargs):
            return LocalizationPipeline(provider=provider).run(**kwargs)

    monkeypatch.setattr("dub_mvp.cli.LocalizationPipeline", FakePipeline)
    runner = CliRunner()

    first = runner.invoke(app, ["localize", str(tmp_path)])
    second = runner.invoke(app, ["localize", str(tmp_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already complete" in second.output
    assert provider.calls == 1

    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.LOCALIZED
    assert loaded.stages["localize"].status == StageStatus.COMPLETED
    assert Path(loaded.outputs["localized_segments"]).is_file()


def test_localize_command_force_reruns_completed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    segments_path = tmp_path / "metadata" / "segments.json"
    segments_path.parent.mkdir()
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["segments"] = str(segments_path)
    manifest.stages["transcribe"].status = StageStatus.COMPLETED
    manifest.save(tmp_path)

    provider = FixtureTranslator()

    class FakePipeline:
        def __init__(self, *, model_name: str) -> None:
            pass

        def run(self, **kwargs):
            return LocalizationPipeline(provider=provider).run(**kwargs)

    monkeypatch.setattr("dub_mvp.cli.LocalizationPipeline", FakePipeline)
    runner = CliRunner()

    first = runner.invoke(app, ["localize", str(tmp_path)])
    second = runner.invoke(app, ["localize", str(tmp_path), "--force"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert provider.calls == 2


def test_localize_command_failure_updates_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    segments_path = tmp_path / "metadata" / "segments.json"
    segments_path.parent.mkdir()
    segments_path.write_text(json.dumps([]))
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["segments"] = str(segments_path)
    manifest.save(tmp_path)

    class FakePipeline:
        def __init__(self, *, model_name: str) -> None:
            pass

        def run(self, **kwargs):
            raise LocalizationError("localization exploded")

    monkeypatch.setattr("dub_mvp.cli.LocalizationPipeline", FakePipeline)
    result = CliRunner().invoke(app, ["localize", str(tmp_path)])

    assert result.exit_code == 1
    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.FAILED
    assert loaded.stages["localize"].status == StageStatus.FAILED
    assert loaded.stages["localize"].error == "localization exploded"

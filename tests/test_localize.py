import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.localize import (
    Glossary,
    LocalizationError,
    LocalizationPipeline,
    NamedEntity,
    OpenAITranslatorProvider,
    TranslationContext,
    TranslationProviderError,
    TranslationProviderResult,
    TranslationUsage,
    build_translation_batches,
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

    translations = {
        "seg_0001": "API deployment demo mein swagat hai.",
        "seg_0002": "Pehle, hum Docker image build karte hain.",
    }

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload
        self.calls = 0
        self.last_glossary_terms: list[str] = []
        self.requests = []

    def localize(self, request):
        assert request.source_language == "en"
        assert request.target_language == "hi"
        self.calls += 1
        self.requests.append(request)
        self.last_glossary_terms = [
            term.source for term in request.glossary.terms
        ]
        payload = self.payload or {
            "batch_id": request.batch_id,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "target_text": self.translations[segment.segment_id],
                    "terms_used": [
                        term.source
                        for term in request.glossary.terms
                        if term.source in segment.source_text
                    ],
                }
                for segment in request.segments
            ],
        }
        return TranslationProviderResult(
            payload=payload,
            usage=TranslationUsage(
                input_tokens=100,
                output_tokens=40,
                cost_usd=0.002,
            ),
        )


class CrashOnSecondBatchTranslator(FixtureTranslator):
    def localize(self, request):
        if request.batch_id == "batch_0002":
            os._exit(23)
        return super().localize(request)


class FailOnceTranslator(FixtureTranslator):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def localize(self, request):
        if request.batch_id == "batch_0002" and not self.failed:
            self.failed = True
            raise TranslationProviderError("provider timeout")
        return super().localize(request)


class GenericLanguageTranslator:
    provider_name = "fixture"
    model_name = "fixture-generic"

    def __init__(self) -> None:
        self.request = None

    def localize(self, request):
        self.request = request
        return TranslationProviderResult(
            payload={
                "batch_id": request.batch_id,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "segments": [
                    {
                        "segment_id": segment.segment_id,
                        "target_text": "Anuvaad",
                    }
                    for segment in request.segments
                ],
            }
        )


def run_crashing_localization(run_directory: str, segments_path: str) -> None:
    LocalizationPipeline(
        provider=CrashOnSecondBatchTranslator(),
        max_batch_utterances=1,
    ).run(
        segments_path=Path(segments_path),
        run_directory=Path(run_directory),
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )


def test_loads_glossary_fixture() -> None:
    glossary = load_glossary(GLOSSARY)

    assert [term.source for term in glossary.terms] == ["Docker", "API"]
    assert glossary.terms[0].target == "Docker"


def test_builds_bounded_batches_with_read_only_neighbor_context() -> None:
    segments = load_transcript_segments(SEGMENTS)
    context = TranslationContext(
        tone="calm technical explanation",
        named_entities=[NamedEntity(source="OpenAI")],
        terminology=[Glossary.model_validate_json(
            GLOSSARY.read_text(encoding="utf-8")
        ).terms[0]],
    )

    batches = build_translation_batches(
        segments,
        source_language="en",
        target_language="hi",
        glossary=load_glossary(GLOSSARY),
        context=context,
        max_batch_utterances=1,
    )

    assert [len(batch.segments) for batch in batches] == [1, 1]
    assert batches[0].preceding_context is None
    assert batches[0].following_context == segments[1].source_text
    assert batches[1].preceding_context == segments[0].source_text
    assert batches[1].following_context is None
    assert batches[0].tone == "calm technical explanation"
    assert batches[0].named_entities[0].source == "OpenAI"
    assert [item.segment_id for item in batches[0].segments] == ["seg_0001"]


def test_validates_localized_segments_in_source_order() -> None:
    source_segments = load_transcript_segments(SEGMENTS)
    localized = validate_localized_segments(
        source_segments,
        {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "target_text": "API deployment demo mein swagat hai.",
                },
                {
                    "segment_id": "seg_0002",
                    "target_text": "Pehle, hum Docker image build karte hain.",
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


def test_rejects_reordered_translator_output() -> None:
    source_segments = load_transcript_segments(SEGMENTS)

    with pytest.raises(LocalizationError, match="order"):
        validate_localized_segments(
            source_segments,
            {
                "segments": [
                    {"segment_id": "seg_0002", "target_text": "Doosra"},
                    {"segment_id": "seg_0001", "target_text": "Pehla"},
                ]
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_id", "batch_wrong", "batch_id"),
        ("source_language", "fr", "source language"),
        ("target_language", "te", "target language"),
    ],
)
def test_rejects_wrong_batch_or_language(
    field: str,
    value: str,
    message: str,
) -> None:
    source = load_transcript_segments(SEGMENTS)[:1]
    payload = {
        "batch_id": "batch_0001",
        "source_language": "en",
        "target_language": "hi",
        "segments": [
            {"segment_id": "seg_0001", "target_text": "Namaste"}
        ],
    }
    payload[field] = value

    with pytest.raises(LocalizationError, match=message):
        validate_localized_segments(
            source,
            payload,
            expected_batch_id="batch_0001",
            source_language="en",
            target_language="hi",
        )


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
    assert Path(outputs["localized_segments_metadata"]).is_file()
    metrics = json.loads(
        Path(outputs["translation_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics["input_tokens"] == 100
    assert metrics["output_tokens"] == 40
    assert metrics["cost_usd"] == pytest.approx(0.002)


def test_pipeline_passes_language_configuration_without_fixed_pair(
    tmp_path: Path,
) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    provider = GenericLanguageTranslator()

    localized, _, _ = LocalizationPipeline(provider=provider).run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="fr",
        target_language="te",
    )

    assert provider.request.source_language == "fr"
    assert provider.request.target_language == "te"
    assert all(item.target_text == "Anuvaad" for item in localized)


def test_openai_adapter_records_token_usage_and_configured_cost(
    monkeypatch,
) -> None:
    request = build_translation_batches(
        load_transcript_segments(SEGMENTS)[:1],
        source_language="en",
        target_language="hi",
        glossary=Glossary(),
        context=TranslationContext(),
    )[0]
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "batch_id": request.batch_id,
                "source_language": "en",
                "target_language": "hi",
                "segments": [
                    {"segment_id": "seg_0001", "target_text": "Namaste"}
                ],
            }
        ),
        usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
    )
    def create(**_):
        return response

    fake_openai = SimpleNamespace(
        OpenAI=lambda: SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
    )
    monkeypatch.setattr("dub_mvp.localize._load_openai", lambda: fake_openai)

    result = OpenAITranslatorProvider(
        model_name="test-model",
        input_cost_per_million=2,
        output_cost_per_million=8,
    ).localize(request)

    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 500
    assert result.usage.cost_usd == pytest.approx(0.006)


def test_every_batch_receives_persisted_glossary_and_context(
    tmp_path: Path,
) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    context_path = tmp_path / "translation-context.json"
    context_path.write_text(
        TranslationContext(
            tone="formal tutorial",
            named_entities=[NamedEntity(source="OpenAI", target="OpenAI")],
        ).model_dump_json(),
        encoding="utf-8",
    )
    provider = FixtureTranslator()

    _, outputs, _ = LocalizationPipeline(
        provider=provider,
        max_batch_utterances=1,
    ).run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
        context_path=context_path,
    )

    assert len(provider.requests) == 2
    assert all(
        [term.source for term in request.glossary.terms] == ["Docker", "API"]
        for request in provider.requests
    )
    assert all(request.tone == "formal tutorial" for request in provider.requests)
    assert all(
        request.named_entities[0].source == "OpenAI"
        for request in provider.requests
    )
    snapshot = json.loads(
        Path(outputs["translation_context"]).read_text(encoding="utf-8")
    )
    assert snapshot["context"]["tone"] == "formal tutorial"


def test_completed_batches_are_reused_without_provider_calls(
    tmp_path: Path,
) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    provider = FixtureTranslator()
    pipeline = LocalizationPipeline(
        provider=provider,
        max_batch_utterances=1,
    )

    pipeline.run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )
    _, outputs, _ = pipeline.run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )

    assert provider.calls == 2
    metrics = json.loads(
        Path(outputs["translation_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics["provider_calls"] == 0
    assert metrics["reused_batches"] == 2


def test_retry_calls_only_the_failed_batch(tmp_path: Path) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    provider = FailOnceTranslator()
    pipeline = LocalizationPipeline(
        provider=provider,
        max_batch_utterances=1,
    )

    with pytest.raises(TranslationProviderError, match="timeout"):
        pipeline.run(
            segments_path=segments_path,
            run_directory=tmp_path,
            source_language="en",
            target_language="hi",
            glossary_path=GLOSSARY,
        )
    localized, outputs, _ = pipeline.run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )

    assert [item.segment_id for item in localized] == ["seg_0001", "seg_0002"]
    assert provider.calls == 2
    metrics = json.loads(
        Path(outputs["translation_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics["reused_batches"] == 1
    assert metrics["attempt_count"] == 3
    assert metrics["failed_attempts"] == 1
    attempts = sorted((tmp_path / "translation" / "batches").glob(
        "batch_0002-*.attempts.json"
    ))
    history = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in history] == ["failed", "completed"]


def test_corrupt_batch_is_regenerated_without_repeating_valid_batch(
    tmp_path: Path,
) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    first_provider = FixtureTranslator()
    LocalizationPipeline(
        provider=first_provider,
        max_batch_utterances=1,
    ).run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )
    corrupt = sorted((tmp_path / "translation" / "batches").glob(
        "batch_0002-*-r*.json"
    ))[0]
    corrupt.write_text("{}", encoding="utf-8")
    second_provider = FixtureTranslator()

    _, outputs, _ = LocalizationPipeline(
        provider=second_provider,
        max_batch_utterances=1,
    ).run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )

    assert second_provider.calls == 1
    assert second_provider.requests[0].batch_id == "batch_0002"
    metrics = json.loads(
        Path(outputs["translation_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics["reused_batches"] == 1
    assert metrics["regenerated_batches"] == 1


def test_context_change_creates_new_batches_without_overwriting_old_revisions(
    tmp_path: Path,
) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    first_context = tmp_path / "context-first.json"
    second_context = tmp_path / "context-second.json"
    first_context.write_text(
        TranslationContext(tone="formal").model_dump_json(),
        encoding="utf-8",
    )
    second_context.write_text(
        TranslationContext(tone="casual").model_dump_json(),
        encoding="utf-8",
    )
    provider = FixtureTranslator()
    pipeline = LocalizationPipeline(
        provider=provider,
        max_batch_utterances=1,
    )

    pipeline.run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        context_path=first_context,
    )
    pipeline.run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        context_path=second_context,
    )

    assert provider.calls == 4
    artifacts = [
        path
        for path in (tmp_path / "translation" / "batches").glob("*-r*.json")
        if not path.name.endswith(".meta.json")
    ]
    assert len(artifacts) == 4


def test_process_death_resumes_from_verified_batch(tmp_path: Path) -> None:
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(SEGMENTS.read_text(encoding="utf-8"))
    process = multiprocessing.get_context("spawn").Process(
        target=run_crashing_localization,
        args=(str(tmp_path), str(segments_path)),
    )

    process.start()
    process.join(timeout=15)

    assert process.exitcode == 23
    completed = sorted((tmp_path / "translation" / "batches").glob(
        "batch_0001-*-r*.meta.json"
    ))
    assert len(completed) == 1
    provider = FixtureTranslator()
    localized, _, _ = LocalizationPipeline(
        provider=provider,
        max_batch_utterances=1,
    ).run(
        segments_path=segments_path,
        run_directory=tmp_path,
        source_language="en",
        target_language="hi",
        glossary_path=GLOSSARY,
    )
    assert provider.calls == 1
    assert provider.requests[0].batch_id == "batch_0002"
    assert len(localized) == 2


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
    provider.provider_name = "openai"
    provider.model_name = "gpt-5-mini"

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
    provider.provider_name = "openai"
    provider.model_name = "gpt-5-mini"

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
    artifacts = [
        path
        for path in (tmp_path / "translation" / "batches").glob("*-r*.json")
        if not path.name.endswith(".meta.json")
    ]
    assert len(artifacts) == 2


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

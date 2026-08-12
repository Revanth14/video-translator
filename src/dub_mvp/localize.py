from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dub_mvp.artifacts import (
    ArtifactMetadata,
    completed_artifact_metadata,
    fingerprint_inputs,
    relative_artifact_path,
    sha256_file,
    verify_artifact,
    write_artifact_metadata,
)
from dub_mvp.transcribe import TranscriptSegment


PROMPT_VERSION = "semantic_translation_v1"
DEFAULT_MAX_BATCH_UTTERANCES = 12
DEFAULT_MAX_BATCH_CHARACTERS = 6000


class LocalizationError(RuntimeError):
    """A permanent localization input or validation failure."""

    retryable = False


class TranslationProviderError(LocalizationError):
    """A transient provider or network failure that can be retried."""

    retryable = True


class TranslationValidationError(LocalizationError):
    """A provider response that violates the translation contract."""


class GlossaryTerm(BaseModel):
    source: str
    target: str
    note: str | None = None

    @field_validator("source", "target")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Glossary source and target cannot be empty.")
        return cleaned


class Glossary(BaseModel):
    terms: list[GlossaryTerm] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_sources(self) -> "Glossary":
        sources = [term.source.casefold() for term in self.terms]
        if len(sources) != len(set(sources)):
            raise ValueError("Glossary source terms must be unique.")
        return self


class NamedEntity(BaseModel):
    source: str
    target: str | None = None
    pronunciation: str | None = None
    note: str | None = None

    @field_validator("source")
    @classmethod
    def source_is_required(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Named entity source cannot be empty.")
        return cleaned

    @field_validator("target", "pronunciation", "note")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split()) or None


class TranslationContext(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    tone: str = "natural conversational speech"
    named_entities: list[NamedEntity] = Field(default_factory=list)
    terminology: list[GlossaryTerm] = Field(default_factory=list)

    @field_validator("tone")
    @classmethod
    def tone_is_required(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Translation tone cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def unique_context_keys(self) -> "TranslationContext":
        entity_names = [item.source.casefold() for item in self.named_entities]
        if len(entity_names) != len(set(entity_names)):
            raise ValueError("Named entities must be unique.")
        terminology = [item.source.casefold() for item in self.terminology]
        if len(terminology) != len(set(terminology)):
            raise ValueError("Terminology source terms must be unique.")
        return self


class LocalizedSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    speaker_id: str | None = None
    source_text: str
    target_text: str
    target_text_revision: int = 1
    localization_status: str = "localized"
    localization_notes: list[str] = Field(default_factory=list)
    glossary_terms: list[str] = Field(default_factory=list)

    @field_validator("segment_id", "source_text", "target_text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Localized segment text cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_timestamps(self) -> "LocalizedSegment":
        if self.start_ms < 0:
            raise ValueError("Segment start_ms cannot be negative.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Segment end_ms must be after start_ms.")
        if self.duration_budget_ms != self.end_ms - self.start_ms:
            raise ValueError("Segment duration budget must match timestamps.")
        if self.target_text_revision < 1:
            raise ValueError("Target text revision must be positive.")
        return self


class TranslationUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class TranslationProviderResult(BaseModel):
    payload: dict[str, Any]
    usage: TranslationUsage = Field(default_factory=TranslationUsage)


class TranslationBatchRequest(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    batch_id: str
    source_language: str
    target_language: str
    prompt_version: str
    tone: str
    glossary: Glossary
    named_entities: list[NamedEntity] = Field(default_factory=list)
    terminology: list[GlossaryTerm] = Field(default_factory=list)
    preceding_context: str | None = None
    following_context: str | None = None
    segments: list[TranscriptSegment]

    @field_validator(
        "batch_id",
        "source_language",
        "target_language",
        "prompt_version",
        "tone",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Translation batch fields cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_ownership(self) -> "TranslationBatchRequest":
        if not self.segments:
            raise ValueError("Translation batch must own at least one utterance.")
        identifiers = [segment.segment_id for segment in self.segments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Translation batch owns duplicate utterance IDs.")
        return self


class TranslationAttemptStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationCostStatus(str, Enum):
    REPORTED = "reported"
    PRICING_UNAVAILABLE = "pricing_unavailable"


class TranslationAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    batch_id: str
    status: TranslationAttemptStatus
    started_at: datetime
    completed_at: datetime
    latency_seconds: float = Field(ge=0)
    provider: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    error_class: str | None = None
    error: str | None = None


class TranslationBatchArtifact(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    batch_id: str
    source_language: str
    target_language: str
    prompt_version: str
    provider: str
    model: str
    attempt_number: int = Field(ge=1)
    started_at: datetime
    completed_at: datetime
    latency_seconds: float = Field(ge=0)
    usage: TranslationUsage
    provider_payload: dict[str, Any]
    segments: list[LocalizedSegment]

    @model_validator(mode="after")
    def validate_unique_segments(self) -> "TranslationBatchArtifact":
        identifiers = [segment.segment_id for segment in self.segments]
        if not identifiers:
            raise ValueError("Translation batch artifact cannot be empty.")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Translation batch artifact contains duplicate IDs.")
        return self


class TranslationMetrics(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    provider: str
    model: str
    prompt_version: str
    configuration_fingerprint: str
    batch_count: int = Field(ge=1)
    provider_calls: int = Field(ge=0)
    reused_batches: int = Field(ge=0)
    regenerated_batches: int = Field(ge=0)
    attempt_count: int = Field(ge=1)
    failed_attempts: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    provider_latency_seconds: float = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    cost_status: TranslationCostStatus


class TranslatorProvider(Protocol):
    provider_name: str
    model_name: str

    def localize(
        self,
        request: TranslationBatchRequest,
    ) -> TranslationProviderResult:
        ...


class OpenAITranslatorProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        model_name: str,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.input_cost_per_million = _configured_price(
            input_cost_per_million,
            "VIDEO_TRANSLATOR_OPENAI_INPUT_USD_PER_MILLION",
        )
        self.output_cost_per_million = _configured_price(
            output_cost_per_million,
            "VIDEO_TRANSLATOR_OPENAI_OUTPUT_USD_PER_MILLION",
        )

    def localize(
        self,
        request: TranslationBatchRequest,
    ) -> TranslationProviderResult:
        openai = _load_openai()
        client = openai.OpenAI()
        try:
            response = client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "system",
                        "content": _localization_instructions(),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            request.model_dump(mode="json"),
                            ensure_ascii=True,
                        ),
                    },
                ],
            )
        except Exception as error:  # provider SDK exception hierarchy varies
            raise TranslationProviderError(
                f"Translator request failed: {type(error).__name__}: {error}"
            ) from error

        text = getattr(response, "output_text", None)
        if not text:
            raise TranslationValidationError(
                "Translator returned no output text."
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise TranslationValidationError(
                "Translator returned invalid JSON."
            ) from error

        input_tokens = _usage_value(response, "input_tokens")
        output_tokens = _usage_value(response, "output_tokens")
        cost = _estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )
        return TranslationProviderResult(
            payload=payload,
            usage=TranslationUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            ),
        )


class LocalizationPipeline:
    def __init__(
        self,
        provider: TranslatorProvider | None = None,
        *,
        model_name: str = "gpt-5-mini",
        prompt_version: str = PROMPT_VERSION,
        max_batch_utterances: int = DEFAULT_MAX_BATCH_UTTERANCES,
        max_batch_characters: int = DEFAULT_MAX_BATCH_CHARACTERS,
    ) -> None:
        if max_batch_utterances <= 0 or max_batch_characters <= 0:
            raise ValueError("Translation batch limits must be positive.")
        self._provider = provider or OpenAITranslatorProvider(
            model_name=model_name
        )
        self.prompt_version = prompt_version
        self.max_batch_utterances = max_batch_utterances
        self.max_batch_characters = max_batch_characters

    def run(
        self,
        *,
        segments_path: Path,
        run_directory: Path,
        source_language: str,
        target_language: str,
        glossary_path: Path | None = None,
        context_path: Path | None = None,
        reuse_completed_batches: bool = True,
    ) -> tuple[list[LocalizedSegment], dict[str, str], str]:
        source_segments = load_transcript_segments(segments_path)
        glossary = load_glossary(glossary_path)
        context = load_translation_context(context_path)
        batches = build_translation_batches(
            source_segments,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
            context=context,
            prompt_version=self.prompt_version,
            max_batch_utterances=self.max_batch_utterances,
            max_batch_characters=self.max_batch_characters,
        )

        configuration = {
            "source_segments_sha256": sha256_file(segments_path),
            "source_language": source_language,
            "target_language": target_language,
            "provider": self._provider.provider_name,
            "model": self._provider.model_name,
            "prompt_version": self.prompt_version,
            "max_batch_utterances": self.max_batch_utterances,
            "max_batch_characters": self.max_batch_characters,
            "glossary": glossary.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        configuration_fingerprint = fingerprint_inputs(configuration)
        translation_directory = run_directory / "translation"
        batches_directory = translation_directory / "batches"
        batches_directory.mkdir(parents=True, exist_ok=True)
        context_snapshot = (
            translation_directory
            / f"context-{configuration_fingerprint[:16]}.json"
        )
        _write_json(context_snapshot, configuration)

        artifacts: list[TranslationBatchArtifact] = []
        artifact_paths: list[Path] = []
        attempts: list[TranslationAttempt] = []
        provider_calls = 0
        reused_batches = 0
        regenerated_batches = 0
        for batch in batches:
            inputs = _batch_inputs(batch, self._provider)
            batch_fingerprint = fingerprint_inputs(inputs)
            stem = f"{batch.batch_id}-{batch_fingerprint[:16]}"
            request_path = batches_directory / f"{stem}.request.json"
            attempts_path = batches_directory / f"{stem}.attempts.json"

            existing, existing_path, invalidated = _find_reusable_batch(
                batches_directory=batches_directory,
                stem=stem,
                expected_inputs=inputs,
                request=batch,
                root=run_directory,
            )
            if existing is not None and existing_path is not None:
                _reconcile_completed_attempt(attempts_path, existing)
            if reuse_completed_batches and existing is not None:
                assert existing_path is not None
                artifacts.append(existing)
                artifact_paths.append(existing_path)
                attempts.extend(_load_attempts(attempts_path))
                reused_batches += 1
                continue
            if existing is not None:
                invalidated = True
            if invalidated:
                regenerated_batches += 1

            _write_json(request_path, batch.model_dump(mode="json"))
            previous_attempts = _load_attempts(attempts_path)
            attempt_number = len(previous_attempts) + 1
            revision = f"r{attempt_number:04d}"
            artifact_path = batches_directory / f"{stem}-{revision}.json"
            metadata_path = batches_directory / f"{stem}-{revision}.meta.json"
            started_at = datetime.now(timezone.utc)
            started = time.monotonic()
            usage = TranslationUsage()
            try:
                result = self._provider.localize(batch)
                if not isinstance(result, TranslationProviderResult):
                    result = TranslationProviderResult.model_validate(result)
                usage = result.usage
                localized = validate_localized_segments(
                    batch.segments,
                    result.payload,
                    expected_batch_id=batch.batch_id,
                    source_language=batch.source_language,
                    target_language=batch.target_language,
                )
            except LocalizationError as error:
                latency = time.monotonic() - started
                _append_attempt(
                    attempts_path,
                    previous_attempts,
                    _failed_attempt(
                        batch=batch,
                        attempt_number=attempt_number,
                        started_at=started_at,
                        latency_seconds=latency,
                        provider=self._provider,
                        usage=usage,
                        error=error,
                    ),
                )
                raise
            except (TypeError, ValueError, ValidationError) as error:
                latency = time.monotonic() - started
                wrapped = TranslationValidationError(
                    f"Invalid translator result: {error}"
                )
                _append_attempt(
                    attempts_path,
                    previous_attempts,
                    _failed_attempt(
                        batch=batch,
                        attempt_number=attempt_number,
                        started_at=started_at,
                        latency_seconds=latency,
                        provider=self._provider,
                        usage=usage,
                        error=wrapped,
                    ),
                )
                raise wrapped from error
            except Exception as error:  # injected providers may raise anything
                latency = time.monotonic() - started
                wrapped = TranslationProviderError(
                    f"Translator request failed: {type(error).__name__}: {error}"
                )
                _append_attempt(
                    attempts_path,
                    previous_attempts,
                    _failed_attempt(
                        batch=batch,
                        attempt_number=attempt_number,
                        started_at=started_at,
                        latency_seconds=latency,
                        provider=self._provider,
                        usage=usage,
                        error=wrapped,
                    ),
                )
                raise wrapped from error

            latency = time.monotonic() - started
            completed_at = datetime.now(timezone.utc)
            artifact = TranslationBatchArtifact(
                batch_id=batch.batch_id,
                source_language=batch.source_language,
                target_language=batch.target_language,
                prompt_version=batch.prompt_version,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                attempt_number=attempt_number,
                started_at=started_at,
                completed_at=completed_at,
                latency_seconds=latency,
                usage=usage,
                provider_payload=result.payload,
                segments=localized,
            )
            _write_json(artifact_path, artifact.model_dump(mode="json"))
            metadata = completed_artifact_metadata(
                artifact_id=batch.batch_id,
                kind="translation_batch",
                path=artifact_path,
                root=run_directory,
                inputs=inputs,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                configuration={
                    "prompt_version": self.prompt_version,
                    "owned_ids": [
                        segment.segment_id for segment in batch.segments
                    ],
                },
            )
            write_artifact_metadata(metadata_path, metadata)
            _append_attempt(
                attempts_path,
                previous_attempts,
                TranslationAttempt(
                    attempt_number=attempt_number,
                    batch_id=batch.batch_id,
                    status=TranslationAttemptStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_seconds=latency,
                    provider=self._provider.provider_name,
                    model=self._provider.model_name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                ),
            )
            attempts.extend(
                [
                    *previous_attempts,
                    TranslationAttempt(
                        attempt_number=attempt_number,
                        batch_id=batch.batch_id,
                        status=TranslationAttemptStatus.COMPLETED,
                        started_at=started_at,
                        completed_at=completed_at,
                        latency_seconds=latency,
                        provider=self._provider.provider_name,
                        model=self._provider.model_name,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    ),
                ]
            )
            artifacts.append(artifact)
            artifact_paths.append(artifact_path)
            provider_calls += 1

        localized_segments = [
            segment for artifact in artifacts for segment in artifact.segments
        ]
        _validate_aggregate(source_segments, localized_segments)
        metrics = _translation_metrics(
            artifacts=artifacts,
            attempts=attempts,
            configuration_fingerprint=configuration_fingerprint,
            provider_calls=provider_calls,
            reused_batches=reused_batches,
            regenerated_batches=regenerated_batches,
        )
        aggregate_inputs = {
            "configuration_fingerprint": configuration_fingerprint,
            "batch_sha256": [sha256_file(path) for path in artifact_paths],
        }
        aggregate_fingerprint = fingerprint_inputs(aggregate_inputs)
        suffix = aggregate_fingerprint[:16]
        localized_path, metadata_path = _find_reusable_aggregate(
            translation_directory=translation_directory,
            suffix=suffix,
            expected_inputs=aggregate_inputs,
            source_segments=source_segments,
            root=run_directory,
        )
        if localized_path is None or metadata_path is None:
            revision = len(
                [
                    path
                    for path in translation_directory.glob(
                        f"localized-{suffix}-r*.json"
                    )
                    if not path.name.endswith(".meta.json")
                ]
            ) + 1
            revision_label = f"r{revision:04d}"
            localized_path = (
                translation_directory
                / f"localized-{suffix}-{revision_label}.json"
            )
            metadata_path = (
                translation_directory
                / f"localized-{suffix}-{revision_label}.meta.json"
            )
            _write_json(
                localized_path,
                [
                    segment.model_dump(mode="json")
                    for segment in localized_segments
                ],
            )
            aggregate_metadata = completed_artifact_metadata(
                artifact_id=f"localized_segments_{revision_label}",
                kind="localized_segments",
                path=localized_path,
                root=run_directory,
                inputs=aggregate_inputs,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                configuration={
                    "prompt_version": self.prompt_version,
                    "configuration_fingerprint": configuration_fingerprint,
                },
            )
            write_artifact_metadata(metadata_path, aggregate_metadata)
        revision_label = localized_path.stem.rsplit("-", 1)[-1]
        summary_path = (
            translation_directory / f"summary-{suffix}-{revision_label}.json"
        )
        if not summary_path.exists():
            _write_json(
                summary_path,
                {
                    "schema_version": 1,
                    "configuration_fingerprint": configuration_fingerprint,
                    "source_language": source_language,
                    "target_language": target_language,
                    "provider": self._provider.provider_name,
                    "model": self._provider.model_name,
                    "prompt_version": self.prompt_version,
                    "batches": [
                        relative_artifact_path(path, run_directory)
                        for path in artifact_paths
                    ],
                },
            )
        metrics_revision = len(
            list(translation_directory.glob(f"metrics-{suffix}-run-*.json"))
        ) + 1
        metrics_path = (
            translation_directory
            / f"metrics-{suffix}-run-{metrics_revision:04d}.json"
        )
        _write_json(metrics_path, metrics.model_dump(mode="json"))

        return localized_segments, {
            "localization_raw": str(summary_path),
            "localized_segments": str(localized_path),
            "localized_segments_metadata": str(metadata_path),
            "translation_context": str(context_snapshot),
            "translation_metrics": str(metrics_path),
        }, self._provider.model_name


def localization_outputs_reusable(
    *,
    outputs: dict[str, str],
    segments_path: Path,
    run_directory: Path,
    source_language: str,
    target_language: str,
    model_name: str,
    provider_name: str = "openai",
    glossary_path: Path | None = None,
    context_path: Path | None = None,
    prompt_version: str = PROMPT_VERSION,
    max_batch_utterances: int = DEFAULT_MAX_BATCH_UTTERANCES,
    max_batch_characters: int = DEFAULT_MAX_BATCH_CHARACTERS,
) -> bool:
    """Prove completed localization outputs still match their current inputs."""
    try:
        source_segments = load_transcript_segments(segments_path)
        glossary = load_glossary(glossary_path)
        context = load_translation_context(context_path)
        batches = build_translation_batches(
            source_segments,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
            context=context,
            prompt_version=prompt_version,
            max_batch_utterances=max_batch_utterances,
            max_batch_characters=max_batch_characters,
        )
        configuration = {
            "source_segments_sha256": sha256_file(segments_path),
            "source_language": source_language,
            "target_language": target_language,
            "provider": provider_name,
            "model": model_name,
            "prompt_version": prompt_version,
            "max_batch_utterances": max_batch_utterances,
            "max_batch_characters": max_batch_characters,
            "glossary": glossary.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        configuration_fingerprint = fingerprint_inputs(configuration)
        artifact_paths: list[Path] = []
        artifacts: list[TranslationBatchArtifact] = []
        batches_directory = run_directory / "translation" / "batches"
        for batch in batches:
            inputs = _batch_input_values(
                batch,
                provider_name=provider_name,
                model_name=model_name,
            )
            batch_fingerprint = fingerprint_inputs(inputs)
            stem = f"{batch.batch_id}-{batch_fingerprint[:16]}"
            artifact, artifact_path, _ = _find_reusable_batch(
                batches_directory=batches_directory,
                stem=stem,
                expected_inputs=inputs,
                request=batch,
                root=run_directory,
            )
            if artifact is None or artifact_path is None:
                return False
            artifacts.append(artifact)
            artifact_paths.append(artifact_path)

        aggregate_inputs = {
            "configuration_fingerprint": configuration_fingerprint,
            "batch_sha256": [sha256_file(path) for path in artifact_paths],
        }
        localized_path = Path(outputs["localized_segments"])
        metadata_path = Path(outputs["localized_segments_metadata"])
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if not verify_artifact(
            metadata,
            expected_inputs=aggregate_inputs,
            root=run_directory,
        ).valid:
            return False
        localized = [
            LocalizedSegment.model_validate(item)
            for item in json.loads(localized_path.read_text(encoding="utf-8"))
        ]
        _validate_aggregate(source_segments, localized)
        return all(
            Path(outputs[name]).is_file()
            for name in (
                "localization_raw",
                "translation_context",
                "translation_metrics",
            )
        )
    except (KeyError, OSError, ValueError, TypeError, LocalizationError):
        return False


def build_translation_batches(
    segments: Sequence[TranscriptSegment],
    *,
    source_language: str,
    target_language: str,
    glossary: Glossary,
    context: TranslationContext,
    prompt_version: str = PROMPT_VERSION,
    max_batch_utterances: int = DEFAULT_MAX_BATCH_UTTERANCES,
    max_batch_characters: int = DEFAULT_MAX_BATCH_CHARACTERS,
) -> list[TranslationBatchRequest]:
    if max_batch_utterances <= 0 or max_batch_characters <= 0:
        raise LocalizationError("Translation batch limits must be positive.")
    if not segments:
        raise LocalizationError("Cannot create translation batches without input.")

    groups: list[tuple[int, int]] = []
    start = 0
    count = 0
    characters = 0
    for index, segment in enumerate(segments):
        segment_characters = len(segment.source_text)
        if segment_characters > max_batch_characters:
            raise LocalizationError(
                f"Utterance {segment.segment_id} exceeds the translation "
                f"batch character limit ({segment_characters} > "
                f"{max_batch_characters})."
            )
        would_overflow = count > 0 and (
            count + 1 > max_batch_utterances
            or characters + segment_characters > max_batch_characters
        )
        if would_overflow:
            groups.append((start, index))
            start = index
            count = 0
            characters = 0
        count += 1
        characters += segment_characters
    groups.append((start, len(segments)))

    return [
        TranslationBatchRequest(
            batch_id=f"batch_{batch_index:04d}",
            source_language=source_language,
            target_language=target_language,
            prompt_version=prompt_version,
            tone=context.tone,
            glossary=glossary,
            named_entities=context.named_entities,
            terminology=context.terminology,
            preceding_context=(
                segments[start - 1].source_text if start > 0 else None
            ),
            following_context=(
                segments[end].source_text if end < len(segments) else None
            ),
            segments=list(segments[start:end]),
        )
        for batch_index, (start, end) in enumerate(groups, start=1)
    ]


def load_transcript_segments(path: Path) -> list[TranscriptSegment]:
    if not path.is_file():
        raise LocalizationError(f"Transcript segments are missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [TranscriptSegment.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError) as error:
        raise LocalizationError(
            f"Unable to read transcript segments: {path}"
        ) from error
    if not segments:
        raise LocalizationError(f"Transcript segments are empty: {path}")
    identifiers = [segment.segment_id for segment in segments]
    if len(identifiers) != len(set(identifiers)):
        raise LocalizationError("Transcript segment IDs must be unique.")
    for previous, current in zip(segments, segments[1:]):
        if current.start_ms < previous.start_ms:
            raise LocalizationError("Transcript segments must be time ordered.")
    return segments


def load_glossary(path: Path | None) -> Glossary:
    if path is None:
        return Glossary()
    if not path.is_file():
        raise LocalizationError(f"Glossary file is missing: {path}")
    try:
        return Glossary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LocalizationError(f"Unable to read glossary: {path}") from error


def load_translation_context(path: Path | None) -> TranslationContext:
    if path is None:
        return TranslationContext()
    if not path.is_file():
        raise LocalizationError(f"Translation context is missing: {path}")
    try:
        return TranslationContext.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise LocalizationError(
            f"Unable to read translation context: {path}"
        ) from error


def validate_localized_segments(
    source_segments: Sequence[TranscriptSegment],
    payload: dict[str, Any],
    *,
    expected_batch_id: str | None = None,
    source_language: str | None = None,
    target_language: str | None = None,
) -> list[LocalizedSegment]:
    if not isinstance(payload, dict):
        raise TranslationValidationError(
            "Translator output must be a JSON object."
        )
    if expected_batch_id is not None and payload.get("batch_id") != expected_batch_id:
        raise TranslationValidationError(
            f"Translator returned the wrong batch_id: {payload.get('batch_id')}."
        )
    if (
        source_language is not None
        and payload.get("source_language") != source_language
    ):
        raise TranslationValidationError(
            "Translator returned the wrong source language."
        )
    if (
        target_language is not None
        and payload.get("target_language") != target_language
    ):
        raise TranslationValidationError(
            "Translator returned the wrong target language."
        )

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise TranslationValidationError(
            "Translator output must contain a 'segments' list."
        )

    expected_ids = [segment.segment_id for segment in source_segments]
    received_ids: list[str] = []
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected = set(expected_ids)
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise TranslationValidationError(
                "Each localized segment must be an object."
            )
        segment_id = str(raw_segment.get("segment_id", ""))
        if segment_id in seen:
            raise TranslationValidationError(
                f"Duplicate segment_id: {segment_id}"
            )
        if segment_id not in expected:
            raise TranslationValidationError(f"Unknown segment_id: {segment_id}")
        seen.add(segment_id)
        received_ids.append(segment_id)
        targets.append(raw_segment)

    missing_ids = expected - seen
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise TranslationValidationError(
            f"Missing localized segment_id: {missing}"
        )
    if received_ids != expected_ids:
        raise TranslationValidationError(
            "Translator changed the owned utterance order."
        )

    localized: list[LocalizedSegment] = []
    for source_segment, raw_target in zip(source_segments, targets):
        try:
            localized.append(
                LocalizedSegment(
                    segment_id=source_segment.segment_id,
                    start_ms=source_segment.start_ms,
                    end_ms=source_segment.end_ms,
                    duration_budget_ms=source_segment.duration_budget_ms,
                    speaker_id=source_segment.speaker_id,
                    source_text=source_segment.source_text,
                    target_text=str(raw_target.get("target_text", "")),
                    target_text_revision=_positive_int(
                        raw_target.get("target_text_revision"),
                        default=1,
                    ),
                    localization_notes=_string_list(raw_target.get("notes")),
                    glossary_terms=_string_list(raw_target.get("terms_used")),
                )
            )
        except ValidationError as error:
            raise TranslationValidationError(
                f"Invalid localized utterance {source_segment.segment_id}: {error}"
            ) from error
    return localized


def _batch_inputs(
    batch: TranslationBatchRequest,
    provider: TranslatorProvider,
) -> dict[str, Any]:
    return _batch_input_values(
        batch,
        provider_name=provider.provider_name,
        model_name=provider.model_name,
    )


def _batch_input_values(
    batch: TranslationBatchRequest,
    *,
    provider_name: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "request": batch.model_dump(mode="json"),
        "provider": provider_name,
        "model": model_name,
    }


def _load_reusable_batch(
    *,
    artifact_path: Path,
    metadata_path: Path,
    expected_inputs: dict[str, Any],
    request: TranslationBatchRequest,
    root: Path,
) -> tuple[TranslationBatchArtifact | None, bool]:
    if not artifact_path.exists() and not metadata_path.exists():
        return None, False
    try:
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        verification = verify_artifact(
            metadata,
            expected_inputs=expected_inputs,
            root=root,
        )
        if not verification.valid:
            return None, True
        artifact = TranslationBatchArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        _validate_batch_artifact(artifact, request)
        return artifact, False
    except (OSError, ValueError, ValidationError, LocalizationError):
        return None, True


def _find_reusable_batch(
    *,
    batches_directory: Path,
    stem: str,
    expected_inputs: dict[str, Any],
    request: TranslationBatchRequest,
    root: Path,
) -> tuple[TranslationBatchArtifact | None, Path | None, bool]:
    metadata_paths = sorted(
        batches_directory.glob(f"{stem}-r*.meta.json"),
        reverse=True,
    )
    invalidated = False
    for metadata_path in metadata_paths:
        artifact_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".json"
        )
        artifact, invalid = _load_reusable_batch(
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            expected_inputs=expected_inputs,
            request=request,
            root=root,
        )
        invalidated = invalidated or invalid
        if artifact is not None:
            return artifact, artifact_path, invalidated
    return None, None, invalidated


def _find_reusable_aggregate(
    *,
    translation_directory: Path,
    suffix: str,
    expected_inputs: dict[str, Any],
    source_segments: Sequence[TranscriptSegment],
    root: Path,
) -> tuple[Path | None, Path | None]:
    metadata_paths = sorted(
        translation_directory.glob(f"localized-{suffix}-r*.meta.json"),
        reverse=True,
    )
    for metadata_path in metadata_paths:
        localized_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".json"
        )
        try:
            metadata = ArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            if not verify_artifact(
                metadata,
                expected_inputs=expected_inputs,
                root=root,
            ).valid:
                continue
            localized = [
                LocalizedSegment.model_validate(item)
                for item in json.loads(
                    localized_path.read_text(encoding="utf-8")
                )
            ]
            _validate_aggregate(source_segments, localized)
            return localized_path, metadata_path
        except (OSError, ValueError, TypeError, LocalizationError):
            continue
    return None, None


def _validate_batch_artifact(
    artifact: TranslationBatchArtifact,
    request: TranslationBatchRequest,
) -> None:
    if (
        artifact.batch_id != request.batch_id
        or artifact.source_language != request.source_language
        or artifact.target_language != request.target_language
        or artifact.prompt_version != request.prompt_version
    ):
        raise LocalizationError("Translation batch artifact identity changed.")
    expected_ids = [segment.segment_id for segment in request.segments]
    actual_ids = [segment.segment_id for segment in artifact.segments]
    if actual_ids != expected_ids:
        raise LocalizationError("Translation batch artifact ownership changed.")
    for source, localized in zip(request.segments, artifact.segments):
        if (
            localized.source_text != source.source_text
            or localized.start_ms != source.start_ms
            or localized.end_ms != source.end_ms
        ):
            raise LocalizationError("Translation batch source provenance changed.")


def _load_attempts(path: Path) -> list[TranslationAttempt]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        attempts = [TranslationAttempt.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError) as error:
        raise LocalizationError(
            f"Translation attempt history is corrupt: {path}"
        ) from error
    expected_numbers = list(range(1, len(attempts) + 1))
    if [attempt.attempt_number for attempt in attempts] != expected_numbers:
        raise LocalizationError(
            f"Translation attempt history is not contiguous: {path}"
        )
    return attempts


def _append_attempt(
    path: Path,
    previous: list[TranslationAttempt],
    attempt: TranslationAttempt,
) -> None:
    if attempt.attempt_number != len(previous) + 1:
        raise LocalizationError("Translation attempt number is not contiguous.")
    _write_json(
        path,
        [item.model_dump(mode="json") for item in [*previous, attempt]],
    )


def _reconcile_completed_attempt(
    path: Path,
    artifact: TranslationBatchArtifact,
) -> None:
    """Close the narrow sidecar-to-attempt-history crash window."""
    attempts = _load_attempts(path)
    if len(attempts) >= artifact.attempt_number:
        return
    if len(attempts) + 1 != artifact.attempt_number:
        raise LocalizationError(
            "Translation artifact and attempt history are inconsistent."
        )
    _append_attempt(
        path,
        attempts,
        TranslationAttempt(
            attempt_number=artifact.attempt_number,
            batch_id=artifact.batch_id,
            status=TranslationAttemptStatus.COMPLETED,
            started_at=artifact.started_at,
            completed_at=artifact.completed_at,
            latency_seconds=artifact.latency_seconds,
            provider=artifact.provider,
            model=artifact.model,
            input_tokens=artifact.usage.input_tokens,
            output_tokens=artifact.usage.output_tokens,
            cost_usd=artifact.usage.cost_usd,
        ),
    )


def _failed_attempt(
    *,
    batch: TranslationBatchRequest,
    attempt_number: int,
    started_at: datetime,
    latency_seconds: float,
    provider: TranslatorProvider,
    usage: TranslationUsage,
    error: LocalizationError,
) -> TranslationAttempt:
    return TranslationAttempt(
        attempt_number=attempt_number,
        batch_id=batch.batch_id,
        status=TranslationAttemptStatus.FAILED,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        latency_seconds=latency_seconds,
        provider=provider.provider_name,
        model=provider.model_name,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        error_class=type(error).__name__,
        error=str(error),
    )


def _validate_aggregate(
    source_segments: Sequence[TranscriptSegment],
    localized_segments: Sequence[LocalizedSegment],
) -> None:
    expected = [segment.segment_id for segment in source_segments]
    actual = [segment.segment_id for segment in localized_segments]
    if actual != expected:
        raise LocalizationError(
            "Translation batches produced missing, duplicate, or reordered "
            "utterances."
        )


def _translation_metrics(
    *,
    artifacts: Sequence[TranslationBatchArtifact],
    attempts: Sequence[TranslationAttempt],
    configuration_fingerprint: str,
    provider_calls: int,
    reused_batches: int,
    regenerated_batches: int,
) -> TranslationMetrics:
    costs = [attempt.cost_usd for attempt in attempts]
    cost_available = all(cost is not None for cost in costs)
    return TranslationMetrics(
        provider=artifacts[0].provider,
        model=artifacts[0].model,
        prompt_version=artifacts[0].prompt_version,
        configuration_fingerprint=configuration_fingerprint,
        batch_count=len(artifacts),
        provider_calls=provider_calls,
        reused_batches=reused_batches,
        regenerated_batches=regenerated_batches,
        attempt_count=len(attempts),
        failed_attempts=sum(
            attempt.status == TranslationAttemptStatus.FAILED
            for attempt in attempts
        ),
        input_tokens=sum(item.input_tokens for item in attempts),
        output_tokens=sum(item.output_tokens for item in attempts),
        provider_latency_seconds=sum(item.latency_seconds for item in attempts),
        cost_usd=(
            sum(cost for cost in costs if cost is not None)
            if cost_available
            else None
        ),
        cost_status=(
            TranslationCostStatus.REPORTED
            if cost_available
            else TranslationCostStatus.PRICING_UNAVAILABLE
        ),
    )


def _load_openai() -> Any:
    try:
        return importlib.import_module("openai")
    except ImportError as error:
        raise LocalizationError(
            "The OpenAI Python package is not installed. Install it in the "
            "runtime that will run 'dub-mvp localize'."
        ) from error


def _localization_instructions() -> str:
    return (
        "Perform semantic translation using the source_language and "
        "target_language in the request. Translate only the owned segments; "
        "preceding_context and following_context are read-only context. Return "
        "only JSON containing batch_id, source_language, target_language, and "
        "a segments array in the exact owned order. Preserve every segment_id "
        "exactly, names, numbers, commands, and technical claims. Apply the "
        "provided tone, glossary, named entities, and terminology. Do not add "
        "facts and do not rewrite merely to fit a duration; timing correction "
        "is a separate stage."
    )


def _usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, 0) if usage is not None else 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _configured_price(value: float | None, environment_name: str) -> float | None:
    if value is not None:
        if value < 0:
            raise ValueError("Translation token prices cannot be negative.")
        return value
    configured = os.getenv(environment_name)
    if not configured:
        return None
    try:
        parsed = float(configured)
    except ValueError as error:
        raise ValueError(f"{environment_name} must be a number.") from error
    if parsed < 0:
        raise ValueError(f"{environment_name} cannot be negative.")
    return parsed


def _estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    return (
        input_tokens * input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

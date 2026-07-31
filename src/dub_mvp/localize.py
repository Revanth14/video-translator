from __future__ import annotations

import importlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dub_mvp.transcribe import TranscriptSegment


class LocalizationError(RuntimeError):
    pass


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

    @field_validator("source_text", "target_text")
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


class TranslatorProvider(Protocol):
    provider_name: str
    model_name: str

    def localize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        source_language: str,
        target_language: str,
        glossary: Glossary,
    ) -> dict[str, Any]:
        ...


class OpenAITranslatorProvider:
    provider_name = "openai"

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def localize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        source_language: str,
        target_language: str,
        glossary: Glossary,
    ) -> dict[str, Any]:
        openai = _load_openai()
        client = openai.OpenAI()
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
                        {
                            "source_language": source_language,
                            "target_language": target_language,
                            "glossary": glossary.model_dump(mode="json"),
                            "segments": [
                                segment.model_dump(mode="json")
                                for segment in segments
                            ],
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        )
        text = getattr(response, "output_text", None)
        if not text:
            raise LocalizationError("Translator returned no output text.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise LocalizationError(
                "Translator returned invalid JSON."
            ) from error


class LocalizationPipeline:
    def __init__(
        self,
        provider: TranslatorProvider | None = None,
        *,
        model_name: str = "gpt-5-mini",
    ) -> None:
        self._provider = provider or OpenAITranslatorProvider(
            model_name=model_name
        )

    def run(
        self,
        *,
        segments_path: Path,
        run_directory: Path,
        source_language: str,
        target_language: str,
        glossary_path: Path | None = None,
    ) -> tuple[list[LocalizedSegment], dict[str, str], str]:
        source_segments = load_transcript_segments(segments_path)
        glossary = load_glossary(glossary_path)

        metadata_directory = run_directory / "metadata"
        metadata_directory.mkdir(parents=True, exist_ok=True)

        raw_payload = self._provider.localize(
            source_segments,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
        )
        raw_path = metadata_directory / "localization_raw.json"
        _write_json(raw_path, raw_payload)

        try:
            localized_segments = validate_localized_segments(
                source_segments,
                raw_payload,
            )
        except (LocalizationError, ValidationError) as error:
            raise LocalizationError(
                f"Invalid localized segments: {error}"
            ) from error
        localized_path = metadata_directory / "localized_segments.json"
        _write_json(
            localized_path,
            [segment.model_dump(mode="json") for segment in localized_segments],
        )

        return localized_segments, {
            "localization_raw": str(raw_path),
            "localized_segments": str(localized_path),
        }, self._provider.model_name


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


def validate_localized_segments(
    source_segments: Sequence[TranscriptSegment],
    payload: dict[str, Any],
) -> list[LocalizedSegment]:
    if not isinstance(payload, dict):
        raise LocalizationError("Translator output must be a JSON object.")

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise LocalizationError(
            "Translator output must contain a 'segments' list."
        )

    expected_by_id = {
        segment.segment_id: segment for segment in source_segments
    }
    seen: set[str] = set()
    target_by_id: dict[str, dict[str, Any]] = {}

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise LocalizationError("Each localized segment must be an object.")
        segment_id = str(raw_segment.get("segment_id", ""))
        if segment_id in seen:
            raise LocalizationError(f"Duplicate segment_id: {segment_id}")
        if segment_id not in expected_by_id:
            raise LocalizationError(f"Unknown segment_id: {segment_id}")
        seen.add(segment_id)
        target_by_id[segment_id] = raw_segment

    missing_ids = set(expected_by_id) - seen
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise LocalizationError(f"Missing localized segment_id: {missing}")

    localized: list[LocalizedSegment] = []
    for source_segment in source_segments:
        raw_target = target_by_id[source_segment.segment_id]
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
    return localized


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
        "Translate English transcript segments into natural spoken Hindi. "
        "Return only JSON with a top-level segments array. Preserve each "
        "segment_id exactly. Preserve names, numbers, commands, and technical "
        "claims. Keep common development terms such as API, server, database, "
        "deployment, Docker, React, and Kubernetes in English when natural. "
        "Do not add facts. Prefer concise phrasing that fits the duration "
        "budget."
    )


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
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

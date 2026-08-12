from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class OverlapStatus(str, Enum):
    NONE = "none"
    POSSIBLE = "possible"
    CONFIRMED = "confirmed"
    UNSUPPORTED = "unsupported"


class DubbingUtterance(BaseModel):
    utterance_id: str
    speaker_id: str | None = None
    start_ms: int
    end_ms: int
    available_duration_ms: int
    source_text: str
    source_segment_ids: list[str] = Field(default_factory=list)
    source_word_indexes: list[int] = Field(default_factory=list)
    preceding_context: str | None = None
    following_context: str | None = None
    overlap_status: OverlapStatus = OverlapStatus.NONE
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("utterance_id", "source_text")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Dubbing utterance fields cannot be empty.")
        return cleaned

    @field_validator("preceding_context", "following_context")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None

    @model_validator(mode="after")
    def validate_timing_and_references(self) -> "DubbingUtterance":
        if self.start_ms < 0:
            raise ValueError("Utterance start_ms cannot be negative.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Utterance end_ms must be after start_ms.")
        if self.available_duration_ms != self.end_ms - self.start_ms:
            raise ValueError("Available duration must match utterance timestamps.")
        if any(index < 0 for index in self.source_word_indexes):
            raise ValueError("Source word indexes cannot be negative.")
        if self.source_word_indexes != sorted(set(self.source_word_indexes)):
            raise ValueError("Source word indexes must be unique and ordered.")
        if len(self.source_segment_ids) != len(set(self.source_segment_ids)):
            raise ValueError("Source segment IDs must be unique.")
        return self


class DubbingUtteranceArtifact(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    source_language: str
    source_transcript: str
    configuration_fingerprint: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    utterances: list[DubbingUtterance]

    @field_validator(
        "source_language",
        "source_transcript",
        "configuration_fingerprint",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Utterance artifact fields cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_utterances(self) -> "DubbingUtteranceArtifact":
        if not self.utterances:
            raise ValueError("Utterance artifact must contain at least one item.")
        identifiers = [item.utterance_id for item in self.utterances]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Dubbing utterance IDs must be unique.")

        # Track the furthest end of the unmarked utterances seen so far, not
        # just the previous one: a long utterance can contain several later
        # ones, and comparing pairwise would only catch the first of them.
        previous = self.utterances[0]
        unmarked_end_ms: int | None = None
        for current in self.utterances:
            if current.start_ms < previous.start_ms:
                raise ValueError("Dubbing utterances must be ordered by start time.")
            if current.overlap_status == OverlapStatus.NONE:
                if unmarked_end_ms is not None and current.start_ms < unmarked_end_ms:
                    raise ValueError(
                        "Overlapping utterances must be marked explicitly."
                    )
                unmarked_end_ms = max(unmarked_end_ms or 0, current.end_ms)
            previous = current
        return self


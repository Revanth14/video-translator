from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from dub_mvp.artifacts import (
    completed_artifact_metadata,
    fingerprint_inputs,
    relative_artifact_path,
    sha256_file,
    write_artifact_metadata,
)
from dub_mvp.transcribe import (
    NormalizedTranscript,
    TranscriptSegment,
    TranscriptWord,
    flatten_words,
)


class UtteranceError(RuntimeError):
    pass


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


class UtterancePipeline:
    """Build the stable, speaker-aware unit consumed by localization.

    Transcription segments remain provider-oriented timing chunks. Dubbing
    utterances are the product contract: deterministic IDs, a single speaker,
    a duration budget, traceability to transcript words, and local context.
    """

    algorithm = "speaker_turn_v1"

    def run(
        self,
        *,
        transcript_path: Path,
        segments_path: Path,
        run_directory: Path,
    ) -> tuple[
        DubbingUtteranceArtifact,
        list[TranscriptSegment],
        dict[str, str],
    ]:
        transcript = _load_transcript(transcript_path)
        segments = _load_segments(segments_path)
        inputs = {
            "algorithm": self.algorithm,
            "transcript_sha256": sha256_file(transcript_path),
            "segments_sha256": sha256_file(segments_path),
        }
        fingerprint = fingerprint_inputs(inputs)
        utterances = build_dubbing_utterances(transcript, segments)
        artifact = DubbingUtteranceArtifact(
            source_language=transcript.language,
            source_transcript=relative_artifact_path(
                transcript_path,
                run_directory,
            ),
            configuration_fingerprint=fingerprint,
            utterances=utterances,
        )
        translation_segments = [
            TranscriptSegment(
                segment_id=item.utterance_id,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                duration_budget_ms=item.available_duration_ms,
                speaker_id=item.speaker_id,
                source_text=item.source_text,
                word_indexes=item.source_word_indexes,
            )
            for item in utterances
        ]

        output_directory = run_directory / "utterances"
        output_directory.mkdir(parents=True, exist_ok=True)
        artifact_path = output_directory / "dubbing_utterances.json"
        translation_path = output_directory / "translation_segments.json"
        metadata_path = output_directory / "dubbing_utterances.meta.json"
        _write_json(artifact_path, artifact.model_dump(mode="json"))
        _write_json(
            translation_path,
            [item.model_dump(mode="json") for item in translation_segments],
        )
        metadata = completed_artifact_metadata(
            artifact_id="dubbing_utterances",
            kind="dubbing_utterances",
            path=artifact_path,
            root=run_directory,
            inputs=inputs,
            configuration={"algorithm": self.algorithm},
        )
        write_artifact_metadata(metadata_path, metadata)
        return artifact, translation_segments, {
            "dubbing_utterances": str(artifact_path),
            "translation_segments": str(translation_path),
            "dubbing_utterances_metadata": str(metadata_path),
        }


def build_dubbing_utterances(
    transcript: NormalizedTranscript,
    segments: list[TranscriptSegment],
) -> list[DubbingUtterance]:
    if not segments:
        raise UtteranceError("Transcription segments are empty.")

    words_by_index = dict(flatten_words(transcript))
    drafts: list[DubbingUtterance] = []
    for segment in segments:
        missing = [
            index
            for index in segment.word_indexes
            if index not in words_by_index
        ]
        if missing:
            # Dropping these silently would lose speech with no trace. The
            # transcript and the segments disagree, which is a defect upstream.
            raise UtteranceError(
                f"Segment {segment.segment_id} references transcript words "
                f"that do not exist: {missing}."
            )
        segment_words = [
            (index, words_by_index[index]) for index in segment.word_indexes
        ]
        groups = _speaker_groups(segment_words)
        if not groups:
            drafts.append(
                _utterance_from_segment(
                    segment,
                    utterance_id=f"utt_{len(drafts) + 1:04d}",
                )
            )
            continue

        for group in groups:
            drafts.append(
                _utterance_from_words(
                    segment,
                    group,
                    utterance_id=f"utt_{len(drafts) + 1:04d}",
                )
            )

    return [
        item.model_copy(
            update={
                "preceding_context": (
                    drafts[index - 1].source_text if index > 0 else None
                ),
                "following_context": (
                    drafts[index + 1].source_text
                    if index + 1 < len(drafts)
                    else None
                ),
            }
        )
        for index, item in enumerate(drafts)
    ]


def _speaker_groups(
    words: list[tuple[int, TranscriptWord]],
) -> list[list[tuple[int, TranscriptWord]]]:
    groups: list[list[tuple[int, TranscriptWord]]] = []
    for item in words:
        if groups and item[1].speaker_id != groups[-1][-1][1].speaker_id:
            groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(item)
    return groups


def _utterance_from_words(
    segment: TranscriptSegment,
    words: list[tuple[int, TranscriptWord]],
    *,
    utterance_id: str,
) -> DubbingUtterance:
    start_ms, end_ms = _safe_timing(
        words[0][1].start_ms,
        words[-1][1].end_ms,
        segment,
    )
    confidences = [
        word.confidence for _, word in words if word.confidence is not None
    ]
    return DubbingUtterance(
        utterance_id=utterance_id,
        speaker_id=words[0][1].speaker_id,
        start_ms=start_ms,
        end_ms=end_ms,
        available_duration_ms=end_ms - start_ms,
        source_text=" ".join(word.text for _, word in words),
        source_segment_ids=[segment.segment_id],
        source_word_indexes=[index for index, _ in words],
        confidence=(sum(confidences) / len(confidences) if confidences else None),
    )


def _utterance_from_segment(
    segment: TranscriptSegment,
    *,
    utterance_id: str,
) -> DubbingUtterance:
    return DubbingUtterance(
        utterance_id=utterance_id,
        speaker_id=segment.speaker_id,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        available_duration_ms=segment.duration_budget_ms,
        source_text=segment.source_text,
        source_segment_ids=[segment.segment_id],
        source_word_indexes=segment.word_indexes,
    )


def _safe_timing(
    start_ms: int,
    end_ms: int,
    segment: TranscriptSegment,
) -> tuple[int, int]:
    start_ms = max(segment.start_ms, min(start_ms, segment.end_ms - 1))
    end_ms = min(segment.end_ms, max(end_ms, start_ms + 1))
    return start_ms, end_ms


def _load_transcript(path: Path) -> NormalizedTranscript:
    try:
        with path.open(encoding="utf-8") as handle:
            return NormalizedTranscript.model_validate(json.load(handle))
    except (OSError, ValueError) as error:
        raise UtteranceError(
            f"Unable to read normalized transcript: {error}"
        ) from error


def _load_segments(path: Path) -> list[TranscriptSegment]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return [TranscriptSegment.model_validate(item) for item in payload]
    except (OSError, TypeError, ValueError) as error:
        raise UtteranceError(
            f"Unable to read transcription segments: {error}"
        ) from error


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

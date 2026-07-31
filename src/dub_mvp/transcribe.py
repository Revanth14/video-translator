from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator


class TranscriptionError(RuntimeError):
    pass


class TranscriptWord(BaseModel):
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    speaker_id: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Transcript word text cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_timestamps(self) -> "TranscriptWord":
        if self.start_ms < 0:
            raise ValueError("Word start_ms cannot be negative.")
        if self.end_ms < self.start_ms:
            raise ValueError("Word end_ms cannot be before start_ms.")
        return self


class TranscriptUtterance(BaseModel):
    utterance_id: str
    start_ms: int
    end_ms: int
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)
    speaker_id: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Transcript utterance text cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_timestamps(self) -> "TranscriptUtterance":
        if self.start_ms < 0:
            raise ValueError("Utterance start_ms cannot be negative.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Utterance end_ms must be after start_ms.")
        previous_end = self.start_ms
        for word in self.words:
            if word.start_ms < self.start_ms or word.end_ms > self.end_ms:
                raise ValueError("Word timestamps must stay inside utterance.")
            if word.start_ms < previous_end:
                raise ValueError("Words must be ordered and non-overlapping.")
            previous_end = word.end_ms
        return self


class NormalizedTranscript(BaseModel):
    schema_version: int = 1
    provider: str = "whisperx"
    model: str
    language: str
    duration_ms: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    utterances: list[TranscriptUtterance]

    @model_validator(mode="after")
    def validate_utterances(self) -> "NormalizedTranscript":
        if self.duration_ms <= 0:
            raise ValueError("Transcript duration_ms must be positive.")
        if not self.utterances:
            raise ValueError("Transcript must contain at least one utterance.")
        previous_end = 0
        for utterance in self.utterances:
            if utterance.end_ms > self.duration_ms:
                raise ValueError("Utterance exceeds transcript duration.")
            if utterance.start_ms < previous_end:
                raise ValueError("Utterances must be ordered and non-overlapping.")
            previous_end = utterance.end_ms
        return self


class TranscriptSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    speaker_id: str | None = None
    source_text: str
    word_indexes: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "TranscriptSegment":
        if self.start_ms < 0:
            raise ValueError("Segment start_ms cannot be negative.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Segment end_ms must be after start_ms.")
        if self.duration_budget_ms != self.end_ms - self.start_ms:
            raise ValueError("Segment duration budget must match timestamps.")
        if not self.source_text.strip():
            raise ValueError("Segment source_text cannot be empty.")
        return self


class TranscriptProvider(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        ...


class WhisperXProvider:
    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self._model: Any | None = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        whisperx = _load_whisperx()
        model = self._get_model(whisperx, language)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language,
        )

        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language") or language,
            device=self.device,
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        aligned.setdefault("language", result.get("language") or language)
        aligned.setdefault("duration_ms", duration_ms)
        aligned.setdefault("model", self.model_name)
        return aligned

    def _get_model(self, whisperx: Any, language: str) -> Any:
        if self._model is None:
            self._model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                language=language,
            )
        return self._model


class TranscriptionPipeline:
    def __init__(
        self,
        provider: TranscriptProvider | None = None,
        *,
        model_name: str = "large-v3",
    ) -> None:
        self._provider = provider or WhisperXProvider(model_name=model_name)
        self._model_name = model_name

    def run(
        self,
        *,
        audio_path: Path,
        run_directory: Path,
        language: str,
        duration_ms: int,
    ) -> tuple[NormalizedTranscript, list[TranscriptSegment], dict[str, str]]:
        if not audio_path.is_file():
            raise TranscriptionError(f"Working audio is missing: {audio_path}")

        metadata_directory = run_directory / "metadata"
        metadata_directory.mkdir(parents=True, exist_ok=True)

        raw_payload = self._provider.transcribe(
            audio_path,
            language=language,
            duration_ms=duration_ms,
        )
        raw_path = metadata_directory / "whisperx_raw.json"
        _write_json(raw_path, raw_payload)

        transcript = normalize_whisperx_result(
            raw_payload,
            duration_ms=duration_ms,
            language=language,
            model_name=str(raw_payload.get("model") or self._model_name),
        )
        segments = build_timestamped_segments(transcript)

        transcript_path = metadata_directory / "transcript.json"
        segments_path = metadata_directory / "segments.json"
        _write_json(transcript_path, transcript.model_dump(mode="json"))
        _write_json(
            segments_path,
            [segment.model_dump(mode="json") for segment in segments],
        )

        return transcript, segments, {
            "whisperx_raw": str(raw_path),
            "transcript": str(transcript_path),
            "segments": str(segments_path),
        }


def normalize_whisperx_result(
    payload: dict[str, Any],
    *,
    duration_ms: int,
    language: str,
    model_name: str,
) -> NormalizedTranscript:
    utterances: list[TranscriptUtterance] = []
    previous_end = 0
    for index, raw_segment in enumerate(payload.get("segments") or [], start=1):
        text = _clean_text(str(raw_segment.get("text", "")))
        if not text:
            continue

        words = _normalize_words(raw_segment.get("words") or [], duration_ms)
        start_ms = _seconds_to_ms(raw_segment.get("start"))
        end_ms = _seconds_to_ms(raw_segment.get("end"))
        if words:
            start_ms = words[0].start_ms
            end_ms = words[-1].end_ms
        start_ms = _clamp_ms(start_ms, 0, duration_ms)
        end_ms = _clamp_ms(end_ms, start_ms + 1, duration_ms)
        if start_ms < previous_end:
            start_ms = previous_end
            words = [
                word for word in words if word.start_ms >= start_ms
            ]
        if end_ms <= start_ms:
            continue

        utterances.append(
            TranscriptUtterance(
                utterance_id=f"utt_{len(utterances) + 1:04d}",
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                words=words,
                speaker_id=raw_segment.get("speaker"),
            )
        )
        previous_end = end_ms

    return NormalizedTranscript(
        provider="whisperx",
        model=model_name,
        language=str(payload.get("language") or language),
        duration_ms=duration_ms,
        utterances=utterances,
    )


def build_timestamped_segments(
    transcript: NormalizedTranscript,
    *,
    min_duration_ms: int = 3000,
    max_duration_ms: int = 12000,
    pause_split_ms: int = 700,
) -> list[TranscriptSegment]:
    words = _flatten_words(transcript)
    if words:
        segments = _segments_from_words(
            words,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            pause_split_ms=pause_split_ms,
        )
    else:
        segments = [
            _segment_from_utterances([utterance], index)
            for index, utterance in enumerate(transcript.utterances, start=1)
        ]

    return [
        segment.model_copy(update={"segment_id": f"seg_{index:04d}"})
        for index, segment in enumerate(segments, start=1)
    ]


def _load_whisperx() -> Any:
    try:
        return importlib.import_module("whisperx")
    except ImportError as error:
        raise TranscriptionError(
            "WhisperX is not installed. Install it in the runtime that will run "
            "'dub-mvp transcribe'."
        ) from error


def _normalize_words(
    raw_words: list[dict[str, Any]],
    duration_ms: int,
) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    previous_end = 0
    for raw_word in raw_words:
        text = _clean_word(str(raw_word.get("word", "")))
        if not text:
            continue
        raw_start = raw_word.get("start")
        raw_end = raw_word.get("end")
        if raw_start is None or raw_end is None:
            continue
        start_ms = _clamp_ms(_seconds_to_ms(raw_start), 0, duration_ms)
        end_ms = _clamp_ms(_seconds_to_ms(raw_end), start_ms, duration_ms)
        if start_ms < previous_end:
            start_ms = previous_end
        if end_ms < start_ms:
            end_ms = start_ms
        words.append(
            TranscriptWord(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=_optional_float(raw_word.get("score")),
                speaker_id=raw_word.get("speaker"),
            )
        )
        previous_end = end_ms
    return words


def _segments_from_words(
    words: list[tuple[int, TranscriptWord]],
    *,
    min_duration_ms: int,
    max_duration_ms: int,
    pause_split_ms: int,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    current: list[tuple[int, TranscriptWord]] = []

    for word_index, word in words:
        if current:
            previous_word = current[-1][1]
            duration_with_word = word.end_ms - current[0][1].start_ms
            pause = word.start_ms - previous_word.end_ms
            can_split_on_pause = (
                pause >= pause_split_ms
                and previous_word.end_ms - current[0][1].start_ms
                >= min_duration_ms
            )
            must_split = duration_with_word > max_duration_ms
            if can_split_on_pause or must_split:
                segments.append(_segment_from_words(current, len(segments) + 1))
                current = []

        current.append((word_index, word))
        duration = current[-1][1].end_ms - current[0][1].start_ms
        if duration >= min_duration_ms and _is_boundary_word(word.text):
            segments.append(_segment_from_words(current, len(segments) + 1))
            current = []

    if current:
        if segments and _segment_duration(current) < min_duration_ms:
            previous = segments.pop()
            merged_words = [
                (index, word)
                for index in previous.word_indexes
                for word_index, word in words
                if word_index == index
            ] + current
            segments.append(_segment_from_words(merged_words, len(segments) + 1))
        else:
            segments.append(_segment_from_words(current, len(segments) + 1))

    return segments


def _segment_from_words(
    words: list[tuple[int, TranscriptWord]],
    index: int,
) -> TranscriptSegment:
    first = words[0][1]
    last = words[-1][1]
    source_text = _clean_text(" ".join(word.text for _, word in words))
    speaker_id = first.speaker_id
    if any(word.speaker_id != speaker_id for _, word in words):
        speaker_id = None
    return TranscriptSegment(
        segment_id=f"seg_{index:04d}",
        start_ms=first.start_ms,
        end_ms=last.end_ms,
        duration_budget_ms=last.end_ms - first.start_ms,
        speaker_id=speaker_id,
        source_text=source_text,
        word_indexes=[word_index for word_index, _ in words],
    )


def _segment_from_utterances(
    utterances: list[TranscriptUtterance],
    index: int,
) -> TranscriptSegment:
    start_ms = utterances[0].start_ms
    end_ms = utterances[-1].end_ms
    speaker_id = utterances[0].speaker_id
    if any(utterance.speaker_id != speaker_id for utterance in utterances):
        speaker_id = None
    return TranscriptSegment(
        segment_id=f"seg_{index:04d}",
        start_ms=start_ms,
        end_ms=end_ms,
        duration_budget_ms=end_ms - start_ms,
        speaker_id=speaker_id,
        source_text=_clean_text(" ".join(item.text for item in utterances)),
    )


def _flatten_words(
    transcript: NormalizedTranscript,
) -> list[tuple[int, TranscriptWord]]:
    indexed_words: list[tuple[int, TranscriptWord]] = []
    for utterance in transcript.utterances:
        for word in utterance.words:
            indexed_words.append((len(indexed_words), word))
    return indexed_words


def _segment_duration(words: list[tuple[int, TranscriptWord]]) -> int:
    return words[-1][1].end_ms - words[0][1].start_ms


def _is_boundary_word(text: str) -> bool:
    return text.endswith((".", "?", "!", ";", ":"))


def _seconds_to_ms(value: Any) -> int:
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return 0


def _clamp_ms(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _clean_word(value: str) -> str:
    return " ".join(value.strip().split())


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

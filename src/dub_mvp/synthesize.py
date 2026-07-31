from __future__ import annotations

import importlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from dub_mvp.localize import LocalizedSegment


class SynthesisError(RuntimeError):
    pass


class VoiceReference(BaseModel):
    reference_id: str
    path: str | None = None
    consent: str
    notes: str | None = None

    @field_validator("reference_id", "consent")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Voice reference fields cannot be empty.")
        return cleaned


class SynthesizedSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    source_text: str
    target_text: str
    target_text_revision: int
    tts_audio_path: str
    tts_duration_ms: int
    tts_revision: int = 1
    model: str
    reference_id: str
    seed: int | None = None
    status: str = "synthesized"
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_text", "target_text", "tts_audio_path", "model")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Synthesized segment text cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_timestamps(self) -> "SynthesizedSegment":
        if self.start_ms < 0:
            raise ValueError("Segment start_ms cannot be negative.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Segment end_ms must be after start_ms.")
        if self.duration_budget_ms != self.end_ms - self.start_ms:
            raise ValueError("Segment duration budget must match timestamps.")
        if self.target_text_revision < 1:
            raise ValueError("Target text revision must be positive.")
        if self.tts_revision < 1:
            raise ValueError("TTS revision must be positive.")
        if self.tts_duration_ms <= 0:
            raise ValueError("TTS duration must be positive.")
        return self


class SynthesisResult(BaseModel):
    audio_path: str
    duration_ms: int
    seed: int | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Synthesis audio path cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_duration(self) -> "SynthesisResult":
        if self.duration_ms <= 0:
            raise ValueError("Synthesis duration must be positive.")
        return self


class SpeechProvider(Protocol):
    provider_name: str
    model_name: str

    def synthesize(
        self,
        segment: LocalizedSegment,
        *,
        output_path: Path,
        voice_reference: VoiceReference,
        target_language: str,
        revision: int,
    ) -> SynthesisResult:
        ...


class IndicF5Provider:
    provider_name = "indicf5"

    def __init__(self, *, model_name: str = "ai4bharat/IndicF5") -> None:
        self.model_name = model_name
        self._module: Any | None = None

    def synthesize(
        self,
        segment: LocalizedSegment,
        *,
        output_path: Path,
        voice_reference: VoiceReference,
        target_language: str,
        revision: int,
    ) -> SynthesisResult:
        module = self._load_module()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = module.synthesize(
            text=segment.target_text,
            output_path=str(output_path),
            language=target_language,
            reference_audio=voice_reference.path,
            model_name=self.model_name,
            revision=revision,
        )
        duration_ms = _positive_int(_result_field(result, "duration_ms"))
        if duration_ms is None:
            duration_ms = _wav_duration_ms(output_path)
        return SynthesisResult(
            audio_path=str(output_path),
            duration_ms=duration_ms,
            seed=_result_field(result, "seed"),
        )

    def _load_module(self) -> Any:
        if self._module is None:
            try:
                self._module = importlib.import_module("indicf5")
            except ImportError as error:
                raise SynthesisError(
                    "IndicF5 is not installed. Install it in the runtime that "
                    "will run 'dub-mvp synthesize'."
                ) from error
        return self._module


class SynthesisPipeline:
    def __init__(
        self,
        provider: SpeechProvider | None = None,
        *,
        model_name: str = "ai4bharat/IndicF5",
    ) -> None:
        self._provider = provider or IndicF5Provider(model_name=model_name)

    def run(
        self,
        *,
        localized_segments_path: Path,
        run_directory: Path,
        target_language: str,
        voice_reference_path: Path,
    ) -> tuple[list[SynthesizedSegment], dict[str, str], str]:
        localized_segments = load_localized_segments(localized_segments_path)
        voice_reference = load_voice_reference(voice_reference_path)
        metadata_directory = run_directory / "metadata"
        metadata_directory.mkdir(parents=True, exist_ok=True)

        synthesized: list[SynthesizedSegment] = []
        raw_results: list[dict[str, Any]] = []
        for segment in localized_segments:
            revision = _next_revision(run_directory, segment.segment_id)
            audio_path = (
                run_directory
                / "segments"
                / segment.segment_id
                / f"tts-r{revision}.wav"
            )
            result = self._provider.synthesize(
                segment,
                output_path=audio_path,
                voice_reference=voice_reference,
                target_language=target_language,
                revision=revision,
            )
            resolved_audio = Path(result.audio_path)
            if not resolved_audio.is_file():
                raise SynthesisError(
                    "Speech provider completed but audio is missing: "
                    f"{resolved_audio}"
                )
            if resolved_audio.stat().st_size <= 0:
                raise SynthesisError(
                    "Speech provider created an empty audio file: "
                    f"{resolved_audio}"
                )
            synthesized.append(
                SynthesizedSegment(
                    segment_id=segment.segment_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    duration_budget_ms=segment.duration_budget_ms,
                    source_text=segment.source_text,
                    target_text=segment.target_text,
                    target_text_revision=segment.target_text_revision,
                    tts_audio_path=str(resolved_audio),
                    tts_duration_ms=result.duration_ms,
                    tts_revision=revision,
                    model=self._provider.model_name,
                    reference_id=voice_reference.reference_id,
                    seed=result.seed,
                    notes=result.notes,
                )
            )
            raw_results.append(
                {
                    "segment_id": segment.segment_id,
                    "audio_path": str(resolved_audio),
                    "duration_ms": result.duration_ms,
                    "revision": revision,
                    "seed": result.seed,
                    "notes": result.notes,
                }
            )

        raw_path = metadata_directory / "synthesis_raw.json"
        synthesized_path = metadata_directory / "synthesized_segments.json"
        _write_json(
            raw_path,
            {
                "provider": self._provider.provider_name,
                "model": self._provider.model_name,
                "voice_reference": voice_reference.model_dump(mode="json"),
                "segments": raw_results,
            },
        )
        _write_json(
            synthesized_path,
            [segment.model_dump(mode="json") for segment in synthesized],
        )
        return synthesized, {
            "synthesis_raw": str(raw_path),
            "synthesized_segments": str(synthesized_path),
        }, self._provider.model_name


def load_localized_segments(path: Path) -> list[LocalizedSegment]:
    if not path.is_file():
        raise SynthesisError(f"Localized segments are missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [LocalizedSegment.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError) as error:
        raise SynthesisError(
            f"Unable to read localized segments: {path}"
        ) from error
    if not segments:
        raise SynthesisError(f"Localized segments are empty: {path}")
    return segments


def load_voice_reference(path: Path) -> VoiceReference:
    if not path.is_file():
        raise SynthesisError(f"Voice reference is missing: {path}")
    try:
        reference = VoiceReference.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise SynthesisError(
            f"Unable to read voice reference: {path}"
        ) from error
    if reference.path and not Path(reference.path).expanduser().is_file():
        raise SynthesisError(
            f"Voice reference audio is missing: {reference.path}"
        )
    return reference


def _next_revision(run_directory: Path, segment_id: str) -> int:
    segment_directory = run_directory / "segments" / segment_id
    existing = []
    for path in segment_directory.glob("tts-r*.wav"):
        raw_revision = path.stem.removeprefix("tts-r")
        parsed = _positive_int(raw_revision)
        if parsed is not None:
            existing.append(parsed)
    return max(existing, default=0) + 1


def _result_field(result: Any, key: str) -> Any:
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _wav_duration_ms(path: Path) -> int:
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            frame_rate = handle.getframerate()
    except (OSError, wave.Error) as error:
        raise SynthesisError(
            f"Unable to measure generated WAV duration: {path}"
        ) from error
    if frame_rate <= 0:
        raise SynthesisError(f"Generated WAV has invalid frame rate: {path}")
    return int(round((frames / frame_rate) * 1000))


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

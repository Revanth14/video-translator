from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import wave
from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dub_mvp.artifacts import (
    ArtifactMetadata,
    completed_artifact_metadata,
    fingerprint_inputs,
    relative_artifact_path,
    sha256_file,
    verify_artifact,
    write_artifact_metadata,
)
from dub_mvp.duration import (
    DurationAttempt,
    DurationCorrectionError,
    DurationCorrector,
    DurationFitArtifact,
    DurationFitStatus,
    DurationMetrics,
    DurationStrategy,
    build_duration_metrics,
    load_duration_fit_artifact,
)
from dub_mvp.indicf5 import (
    IndicF5ChunkingError,
    IndicF5DurationError,
    IndicF5ReferenceError,
    indicf5_duration_plan,
    indicf5_text_plan,
    indicf5_text_normalization_policy_version,
    single_text_batch,
    validate_reference_seconds,
)
from dub_mvp.localize import LocalizedSegment

LOGGER = logging.getLogger(__name__)


class SynthesisError(RuntimeError):
    """A permanent synthesis input or configuration failure."""

    retryable = False


class SpeechProviderError(SynthesisError):
    """A provider failure that may succeed on a bounded retry."""

    retryable = True


class SpeechValidationError(SynthesisError):
    """A speech artifact or provider response that violates the contract."""


class VoiceReference(BaseModel):
    reference_id: str
    path: str | None = None
    reference_text: str | None = None
    consent: str
    notes: str | None = None

    @field_validator("reference_id", "consent")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Voice reference fields cannot be empty.")
        return cleaned

    @field_validator("reference_text")
    @classmethod
    def optional_text_must_not_be_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Voice reference transcript cannot be empty.")
        return cleaned


class VoiceCatalog(BaseModel):
    """Ordered stock/reference voices available to a synthesis run."""

    schema_version: int = Field(default=2, ge=1)
    voices: list[VoiceReference]

    @model_validator(mode="after")
    def validate_voices(self) -> "VoiceCatalog":
        if not self.voices:
            raise ValueError("Voice catalog must contain at least one voice.")
        identifiers = [voice.reference_id for voice in self.voices]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Voice catalog reference IDs must be unique.")
        return self


class SpeakerVoiceAssignment(BaseModel):
    speaker_id: str
    reference_id: str


class VoiceCollision(BaseModel):
    """Distinct speakers that were given the same voice."""

    reference_id: str
    speaker_ids: list[str]


class SpeakerVoiceMap(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    provider: str
    model: str
    target_language: str
    configuration_fingerprint: str
    assignments: list[SpeakerVoiceAssignment]

    @model_validator(mode="after")
    def validate_assignments(self) -> "SpeakerVoiceMap":
        speakers = [assignment.speaker_id for assignment in self.assignments]
        if not speakers:
            raise ValueError("Speaker voice map cannot be empty.")
        if len(speakers) != len(set(speakers)):
            raise ValueError("Speaker voice map contains duplicate speakers.")
        return self

    @property
    def voice_collisions(self) -> list[VoiceCollision]:
        """Voices shared by more than one speaker.

        Derived rather than stored so it can never disagree with the
        assignments, and so maps written by earlier builds report correctly.
        """
        grouped: dict[str, list[str]] = {}
        for assignment in self.assignments:
            grouped.setdefault(assignment.reference_id, []).append(
                assignment.speaker_id
            )
        return [
            VoiceCollision(reference_id=reference_id, speaker_ids=speakers)
            for reference_id, speakers in grouped.items()
            if len(speakers) > 1
        ]


class SpeechAttemptStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class SpeechAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    utterance_id: str
    status: SpeechAttemptStatus
    started_at: datetime
    completed_at: datetime
    latency_seconds: float = Field(ge=0)
    provider: str
    model: str
    voice_id: str
    revision: int = Field(ge=1)
    duration_ms: int | None = Field(default=None, ge=1)
    error_class: str | None = None
    error: str | None = None


class SpeechArtifact(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    utterance_id: str
    speaker_id: str
    voice_id: str
    source_text: str
    target_text: str
    target_text_revision: int = Field(ge=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    duration_budget_ms: int = Field(gt=0)
    target_language: str
    provider: str
    model: str
    revision: int = Field(ge=1)
    attempt_number: int = Field(ge=1)
    started_at: datetime
    completed_at: datetime
    latency_seconds: float = Field(ge=0)
    duration_ms: int = Field(gt=0)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    sample_width_bytes: int = Field(gt=0)
    seed: int | None = None
    notes: list[str] = Field(default_factory=list)
    audio: ArtifactMetadata

    @model_validator(mode="after")
    def validate_timing(self) -> "SpeechArtifact":
        if self.end_ms <= self.start_ms:
            raise ValueError("Speech artifact end_ms must follow start_ms.")
        if self.duration_budget_ms != self.end_ms - self.start_ms:
            raise ValueError("Speech duration budget must match timestamps.")
        return self


class SynthesisMetrics(BaseModel):
    schema_version: int = Field(default=2, ge=1)
    provider: str
    model: str
    configuration_fingerprint: str
    utterance_count: int = Field(ge=1)
    speaker_count: int = Field(ge=1)
    voice_count: int = Field(default=1, ge=1)
    voice_collision_count: int = Field(default=0, ge=0)
    provider_calls: int = Field(ge=0)
    reused_utterances: int = Field(ge=0)
    regenerated_utterances: int = Field(ge=0)
    attempt_count: int = Field(ge=1)
    failed_attempts: int = Field(ge=0)
    provider_latency_seconds: float = Field(ge=0)
    generated_duration_ms: int = Field(ge=1)
    fitted_duration_ms: int | None = Field(default=None, ge=1)
    duration_configuration_fingerprint: str | None = None
    duration_within_primary_count: int = Field(default=0, ge=0)
    duration_within_hard_count: int = Field(default=0, ge=0)
    duration_unresolved_count: int = Field(default=0, ge=0)
    duration_rewrite_count: int = Field(default=0, ge=0)
    duration_attempt_count: int = Field(default=0, ge=0)
    duration_provider_calls: int = Field(default=0, ge=0)
    automated_timing_gate_passed: bool | None = None


class SynthesizedSegment(BaseModel):
    schema_version: int = Field(default=3, ge=1)
    segment_id: str
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    speaker_id: str | None = None
    source_text: str
    target_text: str
    target_text_revision: int
    tts_audio_path: str
    tts_duration_ms: int
    tts_revision: int = 1
    model: str
    reference_id: str
    voice_id: str | None = None
    speech_artifact_path: str | None = None
    speech_artifact_metadata_path: str | None = None
    original_tts_audio_path: str | None = None
    original_tts_duration_ms: int | None = Field(default=None, gt=0)
    duration_error_ms: int | None = None
    duration_ratio: float | None = Field(default=None, gt=0)
    duration_status: str = "legacy_unfitted"
    duration_strategy: str | None = None
    duration_correction_path: str | None = None
    duration_correction_metadata_path: str | None = None
    requires_timing_review: bool = False
    seed: int | None = None
    status: str = "synthesized"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_shape(cls, value: Any) -> Any:
        if isinstance(value, dict) and "schema_version" not in value:
            return {**value, "schema_version": 1}
        return value

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
        if self.schema_version >= 3:
            required = (
                self.original_tts_audio_path,
                self.original_tts_duration_ms,
                self.duration_error_ms,
                self.duration_ratio,
                self.duration_strategy,
                self.duration_correction_path,
                self.duration_correction_metadata_path,
            )
            if any(item is None for item in required):
                raise ValueError(
                    "Schema v3 synthesized segments require duration provenance."
                )
            if self.duration_error_ms != self.tts_duration_ms - self.duration_budget_ms:
                raise ValueError("Synthesized duration error is inconsistent.")
            expected_ratio = self.tts_duration_ms / self.duration_budget_ms
            if abs((self.duration_ratio or 0) - expected_ratio) > 1e-6:
                raise ValueError("Synthesized duration ratio is inconsistent.")
            if self.duration_status not in {
                item.value for item in DurationFitStatus
            }:
                raise ValueError("Synthesized duration status is invalid.")
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


class ControlledSpeechProvider(Protocol):
    """Optional provider capability; ordinary SpeechProvider remains valid."""

    def synthesize_controlled(
        self,
        segment: LocalizedSegment,
        *,
        output_path: Path,
        voice_reference: VoiceReference,
        target_language: str,
        revision: int,
        speaking_rate: float,
        pause_scale: float,
    ) -> SynthesisResult:
        ...


class IndicF5Provider:
    provider_name = "indicf5"

    def __init__(
        self,
        *,
        model_name: str = "ai4bharat/IndicF5",
        runtime_python: str | None = None,
        runtime_script: Path | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.model_name = model_name
        self.runtime_python = runtime_python or os.environ.get(
            "VIDEO_TRANSLATOR_INDICF5_PYTHON"
        )
        self.runtime_script = runtime_script or Path(__file__).with_name(
            "indicf5_runtime.py"
        )
        self.timeout_seconds = timeout_seconds

    def synthesize(
        self,
        segment: LocalizedSegment,
        *,
        output_path: Path,
        voice_reference: VoiceReference,
        target_language: str,
        revision: int,
    ) -> SynthesisResult:
        if voice_reference.path is None:
            raise SynthesisError("IndicF5 requires reference audio.")
        if voice_reference.reference_text is None:
            raise SynthesisError(
                "IndicF5 requires an exact transcript for the reference audio."
            )
        if not self.runtime_python:
            raise SynthesisError(
                "Set VIDEO_TRANSLATOR_INDICF5_PYTHON to the isolated IndicF5 "
                "Python executable."
            )

        reference_audio = Path(voice_reference.path)
        reference_seconds = _wav_duration_ms(reference_audio) / 1000
        try:
            validate_reference_seconds(reference_seconds)
        except IndicF5ReferenceError as error:
            raise SynthesisError(f"Unusable IndicF5 reference: {error}") from error
        try:
            text_plan = indicf5_text_plan(
                text=segment.target_text,
                target_language=target_language,
            )
            batches = single_text_batch(text_plan.tts_text)
        except IndicF5ChunkingError as error:
            raise SynthesisError(f"Unsafe IndicF5 text chunking: {error}") from error
        try:
            duration_plan = indicf5_duration_plan(
                reference_text=voice_reference.reference_text,
                reference_seconds=reference_seconds,
                target_text=text_plan.tts_text,
                target_duration_ms=segment.duration_budget_ms,
            )
        except (IndicF5DurationError, IndicF5ReferenceError) as error:
            raise SynthesisError(f"Unusable IndicF5 duration: {error}") from error
        fix_duration_seconds = duration_plan.fix_duration_seconds
        if not duration_plan.scripts_match:
            # Expected in source-clone dubbing: an English reference prompting
            # Hindi. Recorded so it is visible in review, not blocked.
            LOGGER.info(
                "Utterance %s prompts %s output with a %s reference.",
                segment.segment_id,
                duration_plan.target_script or "unknown",
                duration_plan.reference_script or "unknown",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.with_name(f".{output_path.name}.indicf5-request.json")
        response_path = output_path.with_name(f".{output_path.name}.indicf5-response.json")
        request = {
            "schema_version": 4,
            "model": self.model_name,
            "target_language": target_language,
            "translated_text": segment.target_text,
            "tts_text": text_plan.tts_text,
            "text_normalization_policy": text_plan.policy_version,
            "text_batches": batches,
            "output_path": str(output_path.resolve()),
            "reference_audio": str(reference_audio.resolve()),
            "reference_text": voice_reference.reference_text,
            "reference_seconds": round(reference_seconds, 3),
            "target_duration_ms": segment.duration_budget_ms,
            "fix_duration_seconds": round(fix_duration_seconds, 3),
            "revision": revision,
        }
        _write_json(request_path, request)
        environment = os.environ.copy()
        for credential_name in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "OPENAI_API_KEY",
        ):
            environment.pop(credential_name, None)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        try:
            completed = subprocess.run(
                [
                    self.runtime_python,
                    str(self.runtime_script),
                    str(request_path),
                    str(response_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=environment,
            )
        except FileNotFoundError as error:
            raise SynthesisError(
                f"IndicF5 runtime executable is missing: {self.runtime_python}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise SpeechProviderError(
                f"IndicF5 synthesis exceeded {self.timeout_seconds:g} seconds."
            ) from error
        finally:
            try:
                request_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to remove IndicF5 request %s", request_path)

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            error_type = SynthesisError if completed.returncode == 2 else SpeechProviderError
            raise error_type(
                f"IndicF5 runtime failed with exit code {completed.returncode}: {detail}"
            )
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise SpeechProviderError(
                "IndicF5 runtime did not produce a valid response."
            ) from error
        finally:
            try:
                response_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to remove IndicF5 response %s", response_path)

        duration_ms = _positive_int(response.get("duration_ms"))
        if duration_ms is None:
            duration_ms = _wav_duration_ms(output_path)
        return SynthesisResult(
            audio_path=str(output_path),
            duration_ms=duration_ms,
            seed=response.get("seed"),
            notes=[
                "indicf5_chunk_policy=single_batch_v1",
                f"indicf5_batch_count={len(batches)}",
                *text_plan.notes(),
                *duration_plan.notes(),
            ],
        )


class SynthesisPipeline:
    def __init__(
        self,
        provider: SpeechProvider | None = None,
        *,
        model_name: str = "ai4bharat/IndicF5",
        require_distinct_voices: bool = False,
        duration_corrector: DurationCorrector | None = None,
    ) -> None:
        self._provider = provider or IndicF5Provider(model_name=model_name)
        self.require_distinct_voices = require_distinct_voices
        self.duration_corrector = duration_corrector or DurationCorrector()

    def run(
        self,
        *,
        localized_segments_path: Path,
        run_directory: Path,
        target_language: str,
        voice_reference_path: Path,
        reuse_completed_utterances: bool = True,
    ) -> tuple[list[SynthesizedSegment], dict[str, str], str]:
        localized_segments = load_localized_segments(localized_segments_path)
        voice_catalog = load_voice_catalog(voice_reference_path)
        run_directory = run_directory.resolve()
        speech_directory = run_directory / "speech"
        utterances_directory = speech_directory / "utterances"
        utterances_directory.mkdir(parents=True, exist_ok=True)

        voice_map, voice_map_path, voice_map_metadata_path = (
            _load_or_create_voice_map(
                segments=localized_segments,
                catalog=voice_catalog,
                run_directory=run_directory,
                target_language=target_language,
                provider_name=self._provider.provider_name,
                model_name=self._provider.model_name,
            )
        )
        assignments = {
            assignment.speaker_id: assignment.reference_id
            for assignment in voice_map.assignments
        }
        # More speakers than voices means two people share a voice. A listener
        # notices that immediately, but no existing gate catches it, so it is
        # always recorded and can be made fatal.
        collisions = voice_map.voice_collisions
        if collisions and self.require_distinct_voices:
            detail = "; ".join(
                f"{item.reference_id} -> {', '.join(item.speaker_ids)}"
                for item in collisions
            )
            raise SynthesisError(
                "Voice catalog cannot give every speaker a distinct voice: "
                f"{detail}. Add more voices or allow shared voices."
            )
        if collisions:
            LOGGER.warning(
                "Run %s shares %d voice(s) across speakers: %s",
                run_directory.name,
                len(collisions),
                "; ".join(
                    f"{item.reference_id}={item.speaker_ids}"
                    for item in collisions
                ),
            )
        shared_voice_speakers = {
            speaker
            for collision in collisions
            for speaker in collision.speaker_ids
        }
        voices = {voice.reference_id: voice for voice in voice_catalog.voices}
        configuration = {
            "localized_segments_sha256": sha256_file(localized_segments_path),
            "target_language": target_language,
            "provider": self._provider.provider_name,
            "model": self._provider.model_name,
            "voice_map_sha256": sha256_file(voice_map_path),
            **_provider_synthesis_configuration(self._provider.provider_name),
            "duration_correction": (
                self.duration_corrector.configuration_fingerprint
            ),
        }
        configuration_fingerprint = fingerprint_inputs(configuration)

        artifacts: list[SpeechArtifact] = []
        artifact_paths: list[Path] = []
        artifact_metadata_paths: list[Path] = []
        attempts: list[SpeechAttempt] = []
        provider_calls = 0
        reused_utterances = 0
        regenerated_utterances = 0
        for segment in localized_segments:
            speaker_id = _speaker_identity(segment.speaker_id)
            voice = voices[assignments[speaker_id]]
            inputs = _utterance_inputs(
                segment,
                voice=voice,
                target_language=target_language,
                provider_name=self._provider.provider_name,
                model_name=self._provider.model_name,
            )
            utterance_fingerprint = fingerprint_inputs(inputs)
            directory = utterances_directory / _utterance_directory_name(
                segment.segment_id
            )
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"tts-{utterance_fingerprint[:16]}"
            attempts_path = directory / f"{stem}.attempts.json"
            existing, existing_path, existing_metadata_path, invalidated = (
                _find_reusable_speech_artifact(
                    directory=directory,
                    stem=stem,
                    expected_inputs=inputs,
                    segment=segment,
                    speaker_id=speaker_id,
                    voice_id=voice.reference_id,
                    root=run_directory,
                )
            )
            if existing is not None:
                _reconcile_completed_attempt(attempts_path, existing)
            if reuse_completed_utterances and existing is not None:
                assert existing_path is not None
                assert existing_metadata_path is not None
                artifacts.append(existing)
                artifact_paths.append(existing_path)
                artifact_metadata_paths.append(existing_metadata_path)
                attempts.extend(_load_attempts(attempts_path))
                reused_utterances += 1
                continue
            previous_attempts = _load_attempts(attempts_path)
            if invalidated or existing is not None or previous_attempts:
                regenerated_utterances += 1
            attempt_number = len(previous_attempts) + 1
            revision = _next_speech_revision(directory, stem)
            label = f"r{revision:04d}"
            audio_path = directory / f"{stem}-{label}.wav"
            temporary_audio = directory / f".{stem}-{label}.tmp.wav"
            artifact_path = directory / f"{stem}-{label}.result.json"
            artifact_metadata_path = directory / f"{stem}-{label}.result.meta.json"
            audio_metadata_path = directory / f"{stem}-{label}.wav.meta.json"
            if temporary_audio.exists():
                temporary_audio.unlink()
            started_at = datetime.now(timezone.utc)
            started = time.monotonic()
            try:
                raw_result = self._provider.synthesize(
                    segment,
                    output_path=temporary_audio,
                    voice_reference=voice,
                    target_language=target_language,
                    revision=revision,
                )
                result = (
                    raw_result
                    if isinstance(raw_result, SynthesisResult)
                    else SynthesisResult.model_validate(raw_result)
                )
                if Path(result.audio_path).resolve() != temporary_audio.resolve():
                    raise SpeechProviderError(
                        "Speech provider returned an unexpected audio path: "
                        f"{result.audio_path}"
                    )
                if (
                    not temporary_audio.is_file()
                    or temporary_audio.stat().st_size <= 0
                ):
                    raise SpeechProviderError(
                        "Speech provider created missing or empty audio: "
                        f"{temporary_audio}"
                    )
                _fsync_file(temporary_audio)
                _wav_info(temporary_audio, provider_output=True)
                os.replace(temporary_audio, audio_path)
                audio_info = _wav_info(audio_path, provider_output=True)
            except SynthesisError as error:
                latency = time.monotonic() - started
                _append_attempt(
                    attempts_path,
                    previous_attempts,
                    _failed_attempt(
                        segment=segment,
                        voice=voice,
                        attempt_number=attempt_number,
                        revision=revision,
                        started_at=started_at,
                        latency_seconds=latency,
                        provider=self._provider,
                        error=error,
                    ),
                )
                raise
            except (OSError, TypeError, ValueError, ValidationError) as error:
                latency = time.monotonic() - started
                wrapped = SpeechProviderError(
                    f"Invalid speech provider result: {type(error).__name__}: {error}"
                )
                _append_attempt(
                    attempts_path,
                    previous_attempts,
                    _failed_attempt(
                        segment=segment,
                        voice=voice,
                        attempt_number=attempt_number,
                        revision=revision,
                        started_at=started_at,
                        latency_seconds=latency,
                        provider=self._provider,
                        error=wrapped,
                    ),
                )
                raise wrapped from error
            except Exception as error:  # injected providers may raise anything
                latency = time.monotonic() - started
                wrapped = SpeechProviderError(
                    f"Speech provider failed: {type(error).__name__}: {error}"
                )
                _append_attempt(
                    attempts_path,
                    previous_attempts,
                    _failed_attempt(
                        segment=segment,
                        voice=voice,
                        attempt_number=attempt_number,
                        revision=revision,
                        started_at=started_at,
                        latency_seconds=latency,
                        provider=self._provider,
                        error=wrapped,
                    ),
                )
                raise wrapped from error

            latency = time.monotonic() - started
            completed_at = datetime.now(timezone.utc)
            audio_metadata = completed_artifact_metadata(
                artifact_id=f"{segment.segment_id}_audio_{label}",
                kind="speech_audio",
                path=audio_path,
                root=run_directory,
                inputs=inputs,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                configuration={
                    "voice_id": voice.reference_id,
                    "speaker_id": speaker_id,
                    "target_language": target_language,
                },
            )
            write_artifact_metadata(audio_metadata_path, audio_metadata)
            artifact = SpeechArtifact(
                utterance_id=segment.segment_id,
                speaker_id=speaker_id,
                voice_id=voice.reference_id,
                source_text=segment.source_text,
                target_text=segment.target_text,
                target_text_revision=segment.target_text_revision,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                duration_budget_ms=segment.duration_budget_ms,
                target_language=target_language,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                revision=revision,
                attempt_number=attempt_number,
                started_at=started_at,
                completed_at=completed_at,
                latency_seconds=latency,
                duration_ms=audio_info["duration_ms"],
                sample_rate_hz=audio_info["sample_rate_hz"],
                channels=audio_info["channels"],
                sample_width_bytes=audio_info["sample_width_bytes"],
                seed=result.seed,
                notes=result.notes,
                audio=audio_metadata,
            )
            _write_json(artifact_path, artifact.model_dump(mode="json"))
            artifact_metadata = completed_artifact_metadata(
                artifact_id=f"{segment.segment_id}_speech_{label}",
                kind="speech_result",
                path=artifact_path,
                root=run_directory,
                inputs=inputs,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                configuration={
                    "voice_id": voice.reference_id,
                    "audio_metadata": relative_artifact_path(
                        audio_metadata_path, run_directory
                    ),
                },
            )
            write_artifact_metadata(artifact_metadata_path, artifact_metadata)
            completed_attempt = SpeechAttempt(
                attempt_number=attempt_number,
                utterance_id=segment.segment_id,
                status=SpeechAttemptStatus.COMPLETED,
                started_at=started_at,
                completed_at=completed_at,
                latency_seconds=latency,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                voice_id=voice.reference_id,
                revision=revision,
                duration_ms=artifact.duration_ms,
            )
            _append_attempt(attempts_path, previous_attempts, completed_attempt)
            attempts.extend([*previous_attempts, completed_attempt])
            artifacts.append(artifact)
            artifact_paths.append(artifact_path)
            artifact_metadata_paths.append(artifact_metadata_path)
            provider_calls += 1

        duration_artifacts: list[DurationFitArtifact] = []
        duration_artifact_paths: list[Path] = []
        duration_artifact_metadata_paths: list[Path] = []
        controlled_method = getattr(self._provider, "synthesize_controlled", None)
        for segment, artifact, artifact_path in zip(
            localized_segments, artifacts, artifact_paths
        ):
            speaker_id = _speaker_identity(segment.speaker_id)
            voice = voices[assignments[speaker_id]]

            controlled_synthesizer = None
            if callable(controlled_method):
                controlled_synthesizer = (
                    lambda output_path, speaking_rate, pause_scale, attempt_number,
                    current_segment=segment,
                    current_voice=voice,
                    raw_revision=artifact.revision: _run_controlled_synthesis(
                        controlled_method,
                        current_segment,
                        output_path=output_path,
                        voice=current_voice,
                        target_language=target_language,
                        revision=raw_revision + attempt_number,
                        speaking_rate=speaking_rate,
                        pause_scale=pause_scale,
                    )
                )

            rewritten_synthesizer = None
            if self.duration_corrector.rewriter is not None:
                rewritten_synthesizer = (
                    lambda target_text, target_text_revision, output_path,
                    attempt_number,
                    current_segment=segment,
                    current_voice=voice,
                    raw_revision=artifact.revision: _run_rewritten_synthesis(
                        self._provider,
                        current_segment,
                        target_text=target_text,
                        target_text_revision=target_text_revision,
                        output_path=output_path,
                        voice=current_voice,
                        target_language=target_language,
                        revision=raw_revision + attempt_number,
                    )
                )
            raw_audio_metadata_path = artifact_path.with_name(
                artifact_path.name.removesuffix(".result.json")
                + ".wav.meta.json"
            )
            try:
                fit, fit_path, fit_metadata_path = self.duration_corrector.fit(
                    segment=segment,
                    speech_result_path=artifact_path,
                    raw_audio=artifact.audio,
                    raw_audio_metadata_path=raw_audio_metadata_path,
                    run_directory=run_directory,
                    voice_id=artifact.voice_id,
                    provider=artifact.provider,
                    model=artifact.model,
                    controlled_synthesizer=controlled_synthesizer,
                    rewritten_synthesizer=rewritten_synthesizer,
                )
            except DurationCorrectionError as error:
                raise SynthesisError(str(error)) from error
            duration_artifacts.append(fit)
            duration_artifact_paths.append(fit_path)
            duration_artifact_metadata_paths.append(fit_metadata_path)

        synthesized = [
            _synthesized_segment(
                segment,
                artifact,
                duration_artifact,
                artifact_path=artifact_path,
                artifact_metadata_path=metadata_path,
                duration_artifact_path=duration_artifact_path,
                duration_artifact_metadata_path=duration_metadata_path,
                run_directory=run_directory,
                shares_voice=(
                    _speaker_identity(segment.speaker_id)
                    in shared_voice_speakers
                ),
            )
            for (
                segment,
                artifact,
                duration_artifact,
                artifact_path,
                metadata_path,
                duration_artifact_path,
                duration_metadata_path,
            ) in zip(
                localized_segments,
                artifacts,
                duration_artifacts,
                artifact_paths,
                artifact_metadata_paths,
                duration_artifact_paths,
                duration_artifact_metadata_paths,
            )
        ]
        duration_metrics = build_duration_metrics(
            duration_artifacts,
            configuration_fingerprint=(
                self.duration_corrector.configuration_fingerprint
            ),
        )
        metrics = SynthesisMetrics(
            provider=self._provider.provider_name,
            model=self._provider.model_name,
            configuration_fingerprint=configuration_fingerprint,
            utterance_count=len(artifacts),
            speaker_count=len(voice_map.assignments),
            voice_count=len(voice_catalog.voices),
            voice_collision_count=len(collisions),
            provider_calls=(
                provider_calls + duration_metrics.correction_provider_calls
            ),
            reused_utterances=reused_utterances,
            regenerated_utterances=regenerated_utterances,
            attempt_count=len(attempts),
            failed_attempts=sum(
                attempt.status == SpeechAttemptStatus.FAILED
                for attempt in attempts
            ),
            provider_latency_seconds=sum(
                attempt.latency_seconds for attempt in attempts
            )
            + sum(
                attempt.latency_seconds or 0
                for fit in duration_artifacts
                for attempt in fit.attempts
                if attempt.strategy
                in {
                    DurationStrategy.PROVIDER_CONTROLS,
                    DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                }
            ),
            generated_duration_ms=sum(
                artifact.duration_ms for artifact in artifacts
            ),
            fitted_duration_ms=sum(
                artifact.final_duration_ms for artifact in duration_artifacts
            ),
            duration_configuration_fingerprint=(
                duration_metrics.configuration_fingerprint
            ),
            duration_within_primary_count=duration_metrics.within_primary_count,
            duration_within_hard_count=duration_metrics.within_hard_count,
            duration_unresolved_count=duration_metrics.unresolved_count,
            duration_rewrite_count=duration_metrics.rewrite_count,
            duration_attempt_count=duration_metrics.correction_attempt_count,
            duration_provider_calls=duration_metrics.correction_provider_calls,
            automated_timing_gate_passed=(
                duration_metrics.automated_timing_gate_passed
            ),
        )
        outputs = _write_synthesis_aggregate(
            synthesized=synthesized,
            artifacts=artifacts,
            artifact_paths=artifact_paths,
            duration_artifacts=duration_artifacts,
            duration_artifact_paths=duration_artifact_paths,
            voice_map=voice_map,
            voice_map_path=voice_map_path,
            voice_map_metadata_path=voice_map_metadata_path,
            metrics=metrics,
            duration_metrics=duration_metrics,
            run_directory=run_directory,
        )
        return synthesized, outputs, self._provider.model_name


def load_localized_segments(path: Path) -> list[LocalizedSegment]:
    if not path.is_file():
        raise SynthesisError(f"Localized segments are missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [LocalizedSegment.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError) as error:
        raise SynthesisError(f"Unable to read localized segments: {path}") from error
    if not segments:
        raise SynthesisError(f"Localized segments are empty: {path}")
    identifiers = [segment.segment_id for segment in segments]
    if len(identifiers) != len(set(identifiers)):
        raise SynthesisError("Localized segments contain duplicate IDs.")
    return segments


def load_voice_catalog(path: Path) -> VoiceCatalog:
    if not path.is_file():
        raise SynthesisError(f"Voice reference is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "voices" in payload:
            catalog = VoiceCatalog.model_validate(payload)
        else:
            # Migration path for Phase 5/6 runs containing one VoiceReference.
            catalog = VoiceCatalog(voices=[VoiceReference.model_validate(payload)])
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise SynthesisError(f"Unable to read voice reference: {path}") from error

    resolved_voices = []
    for voice in catalog.voices:
        if voice.path is None:
            resolved_voices.append(voice)
            continue
        audio_path = Path(voice.path).expanduser()
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        if not audio_path.is_file() or audio_path.stat().st_size <= 0:
            raise SynthesisError(
                f"Voice reference audio is missing or empty: {voice.path}"
            )
        resolved_voices.append(
            voice.model_copy(update={"path": str(audio_path.resolve())})
        )
    return catalog.model_copy(update={"voices": resolved_voices})


def load_voice_reference(path: Path) -> VoiceReference:
    """Load the first catalog voice for backward-compatible callers."""

    return load_voice_catalog(path).voices[0]


def synthesis_outputs_reusable(
    *,
    outputs: dict[str, str],
    localized_segments_path: Path,
    voice_reference_path: Path,
    run_directory: Path,
    target_language: str,
    provider_name: str,
    model_name: str,
) -> bool:
    required = {
        "synthesis_raw",
        "synthesis_raw_metadata",
        "synthesized_segments",
        "synthesized_segments_metadata",
        "synthesis_metrics",
        "synthesis_metrics_metadata",
        "speaker_voice_map",
        "speaker_voice_map_metadata",
    }
    if not required.issubset(outputs):
        return False
    try:
        segments = load_localized_segments(localized_segments_path)
        catalog = load_voice_catalog(voice_reference_path)
        voice_map, voice_map_path, voice_map_metadata_path = _find_voice_map(
            segments=segments,
            catalog=catalog,
            run_directory=run_directory,
            target_language=target_language,
            provider_name=provider_name,
            model_name=model_name,
        )
        if (
            voice_map is None
            or voice_map_path is None
            or voice_map_metadata_path is None
        ):
            return False
        if Path(outputs["speaker_voice_map"]).resolve() != voice_map_path.resolve():
            return False
        if (
            Path(outputs["speaker_voice_map_metadata"]).resolve()
            != voice_map_metadata_path.resolve()
        ):
            return False
        assignments = {
            assignment.speaker_id: assignment.reference_id
            for assignment in voice_map.assignments
        }
        voices = {voice.reference_id: voice for voice in catalog.voices}
        artifacts: list[SpeechArtifact] = []
        artifact_paths: list[Path] = []
        for segment in segments:
            speaker_id = _speaker_identity(segment.speaker_id)
            voice = voices[assignments[speaker_id]]
            inputs = _utterance_inputs(
                segment,
                voice=voice,
                target_language=target_language,
                provider_name=provider_name,
                model_name=model_name,
            )
            stem = f"tts-{fingerprint_inputs(inputs)[:16]}"
            artifact, artifact_path, _, _ = _find_reusable_speech_artifact(
                directory=(
                    run_directory
                    / "speech"
                    / "utterances"
                    / _utterance_directory_name(segment.segment_id)
                ),
                stem=stem,
                expected_inputs=inputs,
                segment=segment,
                speaker_id=speaker_id,
                voice_id=voice.reference_id,
                root=run_directory,
            )
            if artifact is None or artifact_path is None:
                return False
            artifacts.append(artifact)
            artifact_paths.append(artifact_path)
        configuration = {
            "localized_segments_sha256": sha256_file(
                localized_segments_path
            ),
            "target_language": target_language,
            "provider": provider_name,
            "model": model_name,
            "voice_map_sha256": sha256_file(voice_map_path),
            **_provider_synthesis_configuration(provider_name),
        }
        aggregate_metadata_path = Path(outputs["synthesized_segments_metadata"])
        aggregate_metadata = ArtifactMetadata.model_validate_json(
            aggregate_metadata_path.read_text(encoding="utf-8")
        )
        aggregate_path = Path(outputs["synthesized_segments"])
        if (
            (run_directory / aggregate_metadata.path).resolve()
            != aggregate_path.resolve()
        ):
            return False
        synthesized = [
            SynthesizedSegment.model_validate(item)
            for item in json.loads(aggregate_path.read_text(encoding="utf-8"))
        ]
        if [item.segment_id for item in synthesized] != [
            item.segment_id for item in segments
        ]:
            return False
        if any(item.schema_version >= 3 for item in synthesized):
            configuration["duration_correction"] = (
                DurationCorrector().configuration_fingerprint
            )
        configuration_fingerprint = fingerprint_inputs(configuration)
        duration_artifact_paths: list[Path] | None = None
        if any(item.schema_version >= 3 for item in synthesized):
            duration_required = {
                "duration_corrections",
                "duration_corrections_metadata",
                "duration_metrics",
                "duration_metrics_metadata",
            }
            if not duration_required.issubset(outputs):
                return False
            duration_artifact_paths = []
            for segment, synthesized_segment in zip(segments, synthesized):
                if (
                    synthesized_segment.schema_version < 3
                    or synthesized_segment.duration_correction_path is None
                    or synthesized_segment.duration_correction_metadata_path is None
                ):
                    return False
                fit_path = run_directory / synthesized_segment.duration_correction_path
                fit_metadata_path = (
                    run_directory
                    / synthesized_segment.duration_correction_metadata_path
                )
                if not _duration_fit_pointer_valid(
                    fit_path=fit_path,
                    metadata_path=fit_metadata_path,
                    segment=segment,
                    root=run_directory,
                ):
                    return False
                duration_artifact_paths.append(fit_path)
        aggregate_inputs = _aggregate_inputs(
            artifacts=artifacts,
            artifact_paths=artifact_paths,
            duration_artifact_paths=duration_artifact_paths,
            voice_map_path=voice_map_path,
        )
        if not verify_artifact(
            aggregate_metadata,
            expected_inputs=aggregate_inputs,
            root=run_directory,
        ).valid:
            return False
        metrics_path = Path(outputs["synthesis_metrics"])
        metrics = SynthesisMetrics.model_validate_json(
            metrics_path.read_text(encoding="utf-8")
        )
        if (
            metrics.provider != provider_name
            or metrics.model != model_name
            or metrics.configuration_fingerprint
            != configuration_fingerprint
            or metrics.utterance_count != len(segments)
        ):
            return False
        if not _verify_named_artifact(
            artifact_path=metrics_path,
            metadata_path=Path(outputs["synthesis_metrics_metadata"]),
            expected_inputs=_telemetry_inputs(
                aggregate_inputs, kind="synthesis_metrics"
            ),
            root=run_directory,
        ):
            return False
        if duration_artifact_paths is not None:
            duration_metrics_path = Path(outputs["duration_metrics"])
            duration_metrics = DurationMetrics.model_validate_json(
                duration_metrics_path.read_text(encoding="utf-8")
            )
            if (
                duration_metrics.utterance_count != len(segments)
                or duration_metrics.configuration_fingerprint
                != metrics.duration_configuration_fingerprint
            ):
                return False
            for name, kind in (
                ("duration_metrics", "duration_metrics"),
                ("duration_corrections", "duration_corrections"),
            ):
                if not _verify_named_artifact(
                    artifact_path=Path(outputs[name]),
                    metadata_path=Path(outputs[f"{name}_metadata"]),
                    expected_inputs=_telemetry_inputs(
                        aggregate_inputs, kind=kind
                    ),
                    root=run_directory,
                ):
                    return False
        raw_path = Path(outputs["synthesis_raw"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if [item["utterance_id"] for item in raw["utterances"]] != [
            item.segment_id for item in segments
        ]:
            return False
        if not _verify_named_artifact(
            artifact_path=raw_path,
            metadata_path=Path(outputs["synthesis_raw_metadata"]),
            expected_inputs=_telemetry_inputs(
                aggregate_inputs, kind="synthesis_run"
            ),
            root=run_directory,
        ):
            return False
        return all(Path(outputs[name]).is_file() for name in required)
    except (KeyError, OSError, ValueError, TypeError, SynthesisError):
        return False


def _load_or_create_voice_map(
    *,
    segments: Sequence[LocalizedSegment],
    catalog: VoiceCatalog,
    run_directory: Path,
    target_language: str,
    provider_name: str,
    model_name: str,
) -> tuple[SpeakerVoiceMap, Path, Path]:
    existing, path, metadata_path = _find_voice_map(
        segments=segments,
        catalog=catalog,
        run_directory=run_directory,
        target_language=target_language,
        provider_name=provider_name,
        model_name=model_name,
    )
    if existing is not None and path is not None and metadata_path is not None:
        return existing, path, metadata_path

    speakers = _ordered_speakers(segments)
    inputs = _voice_map_inputs(
        speakers=speakers,
        catalog=catalog,
        target_language=target_language,
        provider_name=provider_name,
        model_name=model_name,
    )
    fingerprint = fingerprint_inputs(inputs)
    directory = run_directory / "speech" / "voice-maps"
    directory.mkdir(parents=True, exist_ok=True)
    revision = _next_json_revision(directory, f"speaker-voice-map-{fingerprint[:16]}")
    label = f"r{revision:04d}"
    path = directory / f"speaker-voice-map-{fingerprint[:16]}-{label}.json"
    metadata_path = (
        directory
        / f"speaker-voice-map-{fingerprint[:16]}-{label}.meta.json"
    )
    voice_map = SpeakerVoiceMap(
        provider=provider_name,
        model=model_name,
        target_language=target_language,
        configuration_fingerprint=fingerprint,
        assignments=[
            SpeakerVoiceAssignment(
                speaker_id=speaker,
                reference_id=catalog.voices[index % len(catalog.voices)].reference_id,
            )
            for index, speaker in enumerate(speakers)
        ],
    )
    _write_json(path, voice_map.model_dump(mode="json"))
    metadata = completed_artifact_metadata(
        artifact_id=f"speaker_voice_map_{label}",
        kind="speaker_voice_map",
        path=path,
        root=run_directory,
        inputs=inputs,
        provider=provider_name,
        model=model_name,
        configuration={"target_language": target_language},
    )
    write_artifact_metadata(metadata_path, metadata)
    return voice_map, path, metadata_path


def _find_voice_map(
    *,
    segments: Sequence[LocalizedSegment],
    catalog: VoiceCatalog,
    run_directory: Path,
    target_language: str,
    provider_name: str,
    model_name: str,
) -> tuple[SpeakerVoiceMap | None, Path | None, Path | None]:
    speakers = _ordered_speakers(segments)
    inputs = _voice_map_inputs(
        speakers=speakers,
        catalog=catalog,
        target_language=target_language,
        provider_name=provider_name,
        model_name=model_name,
    )
    fingerprint = fingerprint_inputs(inputs)
    directory = run_directory / "speech" / "voice-maps"
    voice_ids = {voice.reference_id for voice in catalog.voices}
    for metadata_path in sorted(
        directory.glob(f"speaker-voice-map-{fingerprint[:16]}-r*.meta.json"),
        reverse=True,
    ):
        path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".json"
        )
        try:
            metadata = ArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            if not verify_artifact(
                metadata, expected_inputs=inputs, root=run_directory
            ).valid:
                continue
            voice_map = SpeakerVoiceMap.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if (
                [item.speaker_id for item in voice_map.assignments] != speakers
                or any(
                    item.reference_id not in voice_ids
                    for item in voice_map.assignments
                )
                or voice_map.configuration_fingerprint != fingerprint
            ):
                continue
            return voice_map, path, metadata_path
        except (OSError, ValueError, ValidationError):
            continue
    return None, None, None


def _load_speech_artifact(
    *,
    artifact_path: Path,
    metadata_path: Path,
    expected_inputs: dict[str, Any],
    segment: LocalizedSegment,
    speaker_id: str,
    voice_id: str,
    root: Path,
) -> tuple[SpeechArtifact | None, bool]:
    if not artifact_path.exists() and not metadata_path.exists():
        return None, False
    try:
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if not verify_artifact(
            metadata, expected_inputs=expected_inputs, root=root
        ).valid:
            return None, True
        artifact = SpeechArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        _validate_speech_artifact(
            artifact,
            segment=segment,
            speaker_id=speaker_id,
            voice_id=voice_id,
        )
        audio_verification = verify_artifact(
            artifact.audio,
            expected_inputs=expected_inputs,
            root=root,
        )
        if not audio_verification.valid:
            return None, True
        audio_metadata_path = artifact_path.with_name(
            artifact_path.name.removesuffix(".result.json")
            + ".wav.meta.json"
        )
        audio_metadata = ArtifactMetadata.model_validate_json(
            audio_metadata_path.read_text(encoding="utf-8")
        )
        if audio_metadata != artifact.audio:
            return None, True
        audio_path = root / artifact.audio.path
        audio_info = _wav_info(audio_path, provider_output=False)
        if (
            audio_info["duration_ms"] != artifact.duration_ms
            or audio_info["sample_rate_hz"] != artifact.sample_rate_hz
            or audio_info["channels"] != artifact.channels
            or audio_info["sample_width_bytes"] != artifact.sample_width_bytes
        ):
            return None, True
        return artifact, False
    except (OSError, ValueError, ValidationError, SynthesisError):
        return None, True


def _find_reusable_speech_artifact(
    *,
    directory: Path,
    stem: str,
    expected_inputs: dict[str, Any],
    segment: LocalizedSegment,
    speaker_id: str,
    voice_id: str,
    root: Path,
) -> tuple[SpeechArtifact | None, Path | None, Path | None, bool]:
    invalidated = False
    for metadata_path in sorted(
        directory.glob(f"{stem}-r*.result.meta.json"), reverse=True
    ):
        artifact_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".json"
        )
        artifact, invalid = _load_speech_artifact(
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            expected_inputs=expected_inputs,
            segment=segment,
            speaker_id=speaker_id,
            voice_id=voice_id,
            root=root,
        )
        invalidated = invalidated or invalid
        if artifact is not None:
            return artifact, artifact_path, metadata_path, invalidated
    return None, None, None, invalidated


def _write_synthesis_aggregate(
    *,
    synthesized: Sequence[SynthesizedSegment],
    artifacts: Sequence[SpeechArtifact],
    artifact_paths: Sequence[Path],
    duration_artifacts: Sequence[DurationFitArtifact],
    duration_artifact_paths: Sequence[Path],
    voice_map: SpeakerVoiceMap,
    voice_map_path: Path,
    voice_map_metadata_path: Path,
    metrics: SynthesisMetrics,
    duration_metrics: DurationMetrics,
    run_directory: Path,
) -> dict[str, str]:
    directory = run_directory / "speech"
    inputs = _aggregate_inputs(
        artifacts=artifacts,
        artifact_paths=artifact_paths,
        duration_artifact_paths=duration_artifact_paths,
        voice_map_path=voice_map_path,
    )
    fingerprint = fingerprint_inputs(inputs)
    stem = f"synthesized-{fingerprint[:16]}"
    aggregate_path: Path | None = None
    aggregate_metadata_path: Path | None = None
    for candidate_metadata in sorted(
        directory.glob(f"{stem}-r*.meta.json"), reverse=True
    ):
        candidate = candidate_metadata.with_name(
            candidate_metadata.name.removesuffix(".meta.json") + ".json"
        )
        try:
            metadata = ArtifactMetadata.model_validate_json(
                candidate_metadata.read_text(encoding="utf-8")
            )
            if verify_artifact(
                metadata, expected_inputs=inputs, root=run_directory
            ).valid:
                parsed = [
                    SynthesizedSegment.model_validate(item)
                    for item in json.loads(candidate.read_text(encoding="utf-8"))
                ]
                if [item.segment_id for item in parsed] == [
                    item.segment_id for item in synthesized
                ]:
                    aggregate_path = candidate
                    aggregate_metadata_path = candidate_metadata
                    break
        except (OSError, ValueError, TypeError, ValidationError):
            continue
    if aggregate_path is None or aggregate_metadata_path is None:
        revision = _next_json_revision(directory, stem)
        label = f"r{revision:04d}"
        aggregate_path = directory / f"{stem}-{label}.json"
        aggregate_metadata_path = directory / f"{stem}-{label}.meta.json"
        _write_json(
            aggregate_path,
            [item.model_dump(mode="json") for item in synthesized],
        )
        metadata = completed_artifact_metadata(
            artifact_id=f"synthesized_segments_{label}",
            kind="synthesized_segments",
            path=aggregate_path,
            root=run_directory,
            inputs=inputs,
            provider=metrics.provider,
            model=metrics.model,
            configuration={
                "configuration_fingerprint": metrics.configuration_fingerprint
            },
        )
        write_artifact_metadata(aggregate_metadata_path, metadata)

    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metrics_path = directory / f"synthesis-metrics-{invocation}.json"
    duration_metrics_path = directory / f"duration-metrics-{invocation}.json"
    duration_path = directory / f"duration-corrections-{invocation}.json"
    raw_path = directory / f"synthesis-run-{invocation}.json"
    _write_json(metrics_path, metrics.model_dump(mode="json"))
    _write_json(duration_metrics_path, duration_metrics.model_dump(mode="json"))
    _write_json(
        duration_path,
        [item.model_dump(mode="json") for item in duration_artifacts],
    )
    _write_json(
        raw_path,
        {
            "schema_version": 3,
            "provider": metrics.provider,
            "model": metrics.model,
            "speaker_voice_map": relative_artifact_path(
                voice_map_path, run_directory
            ),
            "assignments": voice_map.model_dump(mode="json")["assignments"],
            "utterances": [
                {
                    "utterance_id": artifact.utterance_id,
                    "speaker_id": artifact.speaker_id,
                    "voice_id": artifact.voice_id,
                    "result": relative_artifact_path(path, run_directory),
                    "audio": artifact.audio.path,
                    "duration_ms": artifact.duration_ms,
                    "revision": artifact.revision,
                    "attempt_number": artifact.attempt_number,
                }
                for artifact, path in zip(artifacts, artifact_paths)
            ],
            "duration_corrections": [
                {
                    "utterance_id": artifact.utterance_id,
                    "result": relative_artifact_path(path, run_directory),
                    "audio": artifact.audio.path,
                    "original_duration_ms": artifact.original_duration_ms,
                    "final_duration_ms": artifact.final_duration_ms,
                    "status": artifact.status.value,
                    "selected_strategy": artifact.selected_strategy.value,
                }
                for artifact, path in zip(
                    duration_artifacts, duration_artifact_paths
                )
            ],
        },
    )
    metrics_metadata_path = metrics_path.with_name(
        metrics_path.name.removesuffix(".json") + ".meta.json"
    )
    raw_metadata_path = raw_path.with_name(
        raw_path.name.removesuffix(".json") + ".meta.json"
    )
    duration_metrics_metadata_path = duration_metrics_path.with_name(
        duration_metrics_path.name.removesuffix(".json") + ".meta.json"
    )
    duration_metadata_path = duration_path.with_name(
        duration_path.name.removesuffix(".json") + ".meta.json"
    )
    write_artifact_metadata(
        metrics_metadata_path,
        completed_artifact_metadata(
            artifact_id=f"synthesis_metrics_{invocation}",
            kind="synthesis_metrics",
            path=metrics_path,
            root=run_directory,
            inputs=_telemetry_inputs(
                inputs, kind="synthesis_metrics"
            ),
            provider=metrics.provider,
            model=metrics.model,
            configuration={
                "configuration_fingerprint": metrics.configuration_fingerprint
            },
        ),
    )
    write_artifact_metadata(
        raw_metadata_path,
        completed_artifact_metadata(
            artifact_id=f"synthesis_run_{invocation}",
            kind="synthesis_run",
            path=raw_path,
            root=run_directory,
            inputs=_telemetry_inputs(inputs, kind="synthesis_run"),
            provider=metrics.provider,
            model=metrics.model,
        ),
    )
    write_artifact_metadata(
        duration_metrics_metadata_path,
        completed_artifact_metadata(
            artifact_id=f"duration_metrics_{invocation}",
            kind="duration_metrics",
            path=duration_metrics_path,
            root=run_directory,
            inputs=_telemetry_inputs(inputs, kind="duration_metrics"),
            provider=metrics.provider,
            model=metrics.model,
            configuration={
                "configuration_fingerprint": (
                    duration_metrics.configuration_fingerprint
                )
            },
        ),
    )
    write_artifact_metadata(
        duration_metadata_path,
        completed_artifact_metadata(
            artifact_id=f"duration_corrections_{invocation}",
            kind="duration_corrections",
            path=duration_path,
            root=run_directory,
            inputs=_telemetry_inputs(inputs, kind="duration_corrections"),
            provider=metrics.provider,
            model=metrics.model,
            configuration={
                "configuration_fingerprint": (
                    duration_metrics.configuration_fingerprint
                )
            },
        ),
    )
    return {
        "synthesis_raw": str(raw_path),
        "synthesis_raw_metadata": str(raw_metadata_path),
        "synthesis_metrics": str(metrics_path),
        "synthesis_metrics_metadata": str(metrics_metadata_path),
        "duration_metrics": str(duration_metrics_path),
        "duration_metrics_metadata": str(duration_metrics_metadata_path),
        "duration_corrections": str(duration_path),
        "duration_corrections_metadata": str(duration_metadata_path),
        "synthesized_segments": str(aggregate_path),
        "synthesized_segments_metadata": str(aggregate_metadata_path),
        "speaker_voice_map": str(voice_map_path),
        "speaker_voice_map_metadata": str(voice_map_metadata_path),
    }


def _synthesized_segment(
    segment: LocalizedSegment,
    artifact: SpeechArtifact,
    duration_artifact: DurationFitArtifact,
    *,
    artifact_path: Path,
    artifact_metadata_path: Path,
    duration_artifact_path: Path,
    duration_artifact_metadata_path: Path,
    run_directory: Path,
    shares_voice: bool = False,
) -> SynthesizedSegment:
    notes = list(artifact.notes)
    if shares_voice:
        notes.append(
            f"voice {artifact.voice_id} is shared with another speaker"
        )
    return SynthesizedSegment(
        schema_version=3,
        segment_id=segment.segment_id,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        duration_budget_ms=segment.duration_budget_ms,
        speaker_id=segment.speaker_id,
        source_text=segment.source_text,
        target_text=duration_artifact.target_text,
        target_text_revision=duration_artifact.target_text_revision,
        tts_audio_path=str(
            (run_directory / duration_artifact.audio.path).resolve()
        ),
        tts_duration_ms=duration_artifact.final_duration_ms,
        tts_revision=artifact.revision,
        model=artifact.model,
        reference_id=artifact.voice_id,
        voice_id=artifact.voice_id,
        speech_artifact_path=relative_artifact_path(artifact_path, run_directory),
        speech_artifact_metadata_path=relative_artifact_path(
            artifact_metadata_path, run_directory
        ),
        original_tts_audio_path=str((run_directory / artifact.audio.path).resolve()),
        original_tts_duration_ms=artifact.duration_ms,
        duration_error_ms=duration_artifact.duration_error_ms,
        duration_ratio=duration_artifact.duration_ratio,
        duration_status=duration_artifact.status.value,
        duration_strategy=duration_artifact.selected_strategy.value,
        duration_correction_path=relative_artifact_path(
            duration_artifact_path, run_directory
        ),
        duration_correction_metadata_path=relative_artifact_path(
            duration_artifact_metadata_path, run_directory
        ),
        requires_timing_review=duration_artifact.needs_human_review,
        seed=artifact.seed,
        notes=notes,
    )


def _voice_map_inputs(
    *,
    speakers: Sequence[str],
    catalog: VoiceCatalog,
    target_language: str,
    provider_name: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "speakers": list(speakers),
        "voices": [_voice_descriptor(voice) for voice in catalog.voices],
        "target_language": target_language,
        "provider": provider_name,
        "model": model_name,
        "assignment_policy": "ordered_round_robin_v1",
    }


def _utterance_inputs(
    segment: LocalizedSegment,
    *,
    voice: VoiceReference,
    target_language: str,
    provider_name: str,
    model_name: str,
) -> dict[str, Any]:
    inputs = {
        "utterance": segment.model_dump(mode="json"),
        "speaker_id": _speaker_identity(segment.speaker_id),
        "voice": _voice_descriptor(voice),
        "target_language": target_language,
        "provider": provider_name,
        "model": model_name,
    }
    if provider_name == "indicf5":
        text_plan = indicf5_text_plan(
            text=segment.target_text,
            target_language=target_language,
        )
        inputs["provider_text"] = {
            "normalization_policy": text_plan.policy_version,
            "tts_text": text_plan.tts_text,
        }
    return inputs


def _provider_synthesis_configuration(provider_name: str) -> dict[str, str]:
    if provider_name == "indicf5":
        return {
            "text_normalization_policy": (
                indicf5_text_normalization_policy_version()
            ),
        }
    return {}


def _voice_descriptor(voice: VoiceReference) -> dict[str, Any]:
    return {
        "reference_id": voice.reference_id,
        "reference_audio_sha256": (
            sha256_file(Path(voice.path)) if voice.path is not None else None
        ),
        "reference_text": voice.reference_text,
    }


def _aggregate_inputs(
    *,
    artifacts: Sequence[SpeechArtifact],
    artifact_paths: Sequence[Path],
    duration_artifact_paths: Sequence[Path] | None = None,
    voice_map_path: Path,
) -> dict[str, Any]:
    inputs = {
        "utterance_ids": [artifact.utterance_id for artifact in artifacts],
        "speech_result_sha256": [sha256_file(path) for path in artifact_paths],
        "voice_map_sha256": sha256_file(voice_map_path),
    }
    if duration_artifact_paths is not None:
        inputs["duration_fit_sha256"] = [
            sha256_file(path) for path in duration_artifact_paths
        ]
    return inputs


def _telemetry_inputs(
    aggregate_inputs: dict[str, Any], *, kind: str
) -> dict[str, Any]:
    return {"aggregate": aggregate_inputs, "kind": kind}


def _verify_named_artifact(
    *,
    artifact_path: Path,
    metadata_path: Path,
    expected_inputs: dict[str, Any],
    root: Path,
) -> bool:
    metadata = ArtifactMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )
    if (root / metadata.path).resolve() != artifact_path.resolve():
        return False
    return verify_artifact(
        metadata, expected_inputs=expected_inputs, root=root
    ).valid


def _duration_fit_pointer_valid(
    *,
    fit_path: Path,
    metadata_path: Path,
    segment: LocalizedSegment,
    root: Path,
) -> bool:
    try:
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if (
            (root / metadata.path).resolve() != fit_path.resolve()
            or metadata.status.value != "completed"
            or not fit_path.is_file()
            or fit_path.stat().st_size != metadata.size_bytes
            or sha256_file(fit_path) != metadata.output_sha256
        ):
            return False
        fit = load_duration_fit_artifact(fit_path)
        if (
            fit.utterance_id != segment.segment_id
            or fit.start_ms != segment.start_ms
            or fit.end_ms != segment.end_ms
            or fit.available_duration_ms != segment.duration_budget_ms
        ):
            return False
        audio_metadata_path = root / fit.audio_metadata_path
        audio_metadata = ArtifactMetadata.model_validate_json(
            audio_metadata_path.read_text(encoding="utf-8")
        )
        audio_path = root / fit.audio.path
        if (
            audio_metadata != fit.audio
            or audio_metadata.status.value != "completed"
            or not audio_path.is_file()
            or audio_path.stat().st_size != audio_metadata.size_bytes
            or sha256_file(audio_path) != audio_metadata.output_sha256
            or _wav_info(audio_path, provider_output=False)["duration_ms"]
            != fit.final_duration_ms
        ):
            return False
        attempts = [
            DurationAttempt.model_validate(item)
            for item in json.loads(
                (root / fit.attempts_path).read_text(encoding="utf-8")
            )
        ]
        return attempts == fit.attempts
    except (
        OSError,
        ValueError,
        TypeError,
        ValidationError,
        SynthesisError,
        DurationCorrectionError,
    ):
        return False


def _validate_speech_artifact(
    artifact: SpeechArtifact,
    *,
    segment: LocalizedSegment,
    speaker_id: str,
    voice_id: str,
) -> None:
    if (
        artifact.utterance_id != segment.segment_id
        or artifact.speaker_id != speaker_id
        or artifact.voice_id != voice_id
        or artifact.source_text != segment.source_text
        or artifact.target_text != segment.target_text
        or artifact.target_text_revision != segment.target_text_revision
        or artifact.start_ms != segment.start_ms
        or artifact.end_ms != segment.end_ms
        or artifact.duration_budget_ms != segment.duration_budget_ms
    ):
        raise SpeechValidationError("Speech artifact provenance changed.")


def _ordered_speakers(segments: Sequence[LocalizedSegment]) -> list[str]:
    speakers: list[str] = []
    for segment in segments:
        speaker = _speaker_identity(segment.speaker_id)
        if speaker not in speakers:
            speakers.append(speaker)
    return speakers


def _speaker_identity(speaker_id: str | None) -> str:
    return speaker_id or "speaker_unknown"


def _utterance_directory_name(utterance_id: str) -> str:
    return f"u-{fingerprint_inputs({'utterance_id': utterance_id})[:16]}"


def _next_speech_revision(directory: Path, stem: str) -> int:
    revisions: list[int] = []
    for path in directory.glob(f"{stem}-r*.wav"):
        raw = path.stem.rsplit("-r", 1)[-1]
        parsed = _positive_int(raw)
        if parsed is not None:
            revisions.append(parsed)
    return max(revisions, default=0) + 1


def _next_json_revision(directory: Path, stem: str) -> int:
    revisions: list[int] = []
    for path in directory.glob(f"{stem}-r*.json"):
        if path.name.endswith(".meta.json"):
            continue
        raw = path.stem.rsplit("-r", 1)[-1]
        parsed = _positive_int(raw)
        if parsed is not None:
            revisions.append(parsed)
    return max(revisions, default=0) + 1


def _load_attempts(path: Path) -> list[SpeechAttempt]:
    if not path.exists():
        return []
    try:
        attempts = [
            SpeechAttempt.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise SynthesisError(f"Speech attempt history is corrupt: {path}") from error
    if [item.attempt_number for item in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise SynthesisError(f"Speech attempt history is not contiguous: {path}")
    return attempts


def _append_attempt(
    path: Path, previous: list[SpeechAttempt], attempt: SpeechAttempt
) -> None:
    if attempt.attempt_number != len(previous) + 1:
        raise SynthesisError("Speech attempt number is not contiguous.")
    _write_json(
        path,
        [item.model_dump(mode="json") for item in [*previous, attempt]],
    )


def _reconcile_completed_attempt(path: Path, artifact: SpeechArtifact) -> None:
    attempts = _load_attempts(path)
    if len(attempts) >= artifact.attempt_number:
        return
    if len(attempts) + 1 != artifact.attempt_number:
        raise SynthesisError("Speech artifact and attempt history are inconsistent.")
    _append_attempt(
        path,
        attempts,
        SpeechAttempt(
            attempt_number=artifact.attempt_number,
            utterance_id=artifact.utterance_id,
            status=SpeechAttemptStatus.COMPLETED,
            started_at=artifact.started_at,
            completed_at=artifact.completed_at,
            latency_seconds=artifact.latency_seconds,
            provider=artifact.provider,
            model=artifact.model,
            voice_id=artifact.voice_id,
            revision=artifact.revision,
            duration_ms=artifact.duration_ms,
        ),
    )


def _failed_attempt(
    *,
    segment: LocalizedSegment,
    voice: VoiceReference,
    attempt_number: int,
    revision: int,
    started_at: datetime,
    latency_seconds: float,
    provider: SpeechProvider,
    error: SynthesisError,
) -> SpeechAttempt:
    return SpeechAttempt(
        attempt_number=attempt_number,
        utterance_id=segment.segment_id,
        status=SpeechAttemptStatus.FAILED,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        latency_seconds=latency_seconds,
        provider=provider.provider_name,
        model=provider.model_name,
        voice_id=voice.reference_id,
        revision=revision,
        error_class=type(error).__name__,
        error=str(error),
    )


def _run_controlled_synthesis(
    method: Any,
    segment: LocalizedSegment,
    *,
    output_path: Path,
    voice: VoiceReference,
    target_language: str,
    revision: int,
    speaking_rate: float,
    pause_scale: float,
) -> None:
    raw = method(
        segment,
        output_path=output_path,
        voice_reference=voice,
        target_language=target_language,
        revision=revision,
        speaking_rate=speaking_rate,
        pause_scale=pause_scale,
    )
    _validate_correction_provider_result(raw, output_path)


def _run_rewritten_synthesis(
    provider: SpeechProvider,
    segment: LocalizedSegment,
    *,
    target_text: str,
    target_text_revision: int,
    output_path: Path,
    voice: VoiceReference,
    target_language: str,
    revision: int,
) -> None:
    rewritten = segment.model_copy(
        update={
            "target_text": target_text,
            "target_text_revision": target_text_revision,
        }
    )
    raw = provider.synthesize(
        rewritten,
        output_path=output_path,
        voice_reference=voice,
        target_language=target_language,
        revision=revision,
    )
    _validate_correction_provider_result(raw, output_path)


def _validate_correction_provider_result(raw: Any, output_path: Path) -> None:
    result = (
        raw if isinstance(raw, SynthesisResult) else SynthesisResult.model_validate(raw)
    )
    if Path(result.audio_path).resolve() != output_path.resolve():
        raise SpeechProviderError(
            "Duration synthesis returned an unexpected audio path: "
            f"{result.audio_path}"
        )


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
    return parsed if parsed > 0 else None


def _wav_info(path: Path, *, provider_output: bool) -> dict[str, int]:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise OSError("file is missing or empty")
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            frame_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
    except (OSError, wave.Error) as error:
        error_type = SpeechProviderError if provider_output else SpeechValidationError
        raise error_type(f"Unable to decode generated WAV audio: {path}") from error
    if frames <= 0 or frame_rate <= 0 or channels <= 0 or sample_width <= 0:
        error_type = SpeechProviderError if provider_output else SpeechValidationError
        raise error_type(f"Generated WAV has invalid audio parameters: {path}")
    return {
        "duration_ms": max(1, int(round((frames / frame_rate) * 1000))),
        "sample_rate_hz": frame_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }


def _wav_duration_ms(path: Path) -> int:
    return _wav_info(path, provider_output=True)["duration_ms"]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

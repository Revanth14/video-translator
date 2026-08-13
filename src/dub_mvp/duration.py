from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import wave
from collections.abc import Callable, Sequence
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
from dub_mvp.localize import LocalizedSegment
from dub_mvp.manifest import redact_sensitive_text


class DurationCorrectionError(RuntimeError):
    """A corrupt input or invalid duration-correction configuration."""

    retryable = False


class DurationStrategy(str, Enum):
    ACCEPT = "accept"
    PROVIDER_CONTROLS = "provider_controls"
    TRIM_ARTIFICIAL_PAUSES = "trim_artificial_pauses"
    MILD_TIME_STRETCH = "mild_time_stretch"
    COMPACT_REWRITE = "compact_rewrite"
    REGENERATE_ASSIGNED_VOICE = "regenerate_assigned_voice"
    FINAL_TIME_STRETCH = "final_time_stretch"
    SURFACE_UNRESOLVED = "surface_unresolved"


class DurationAttemptStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DurationFitStatus(str, Enum):
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REVIEW_REQUIRED = "review_required"
    UNRESOLVED = "unresolved"


class DurationPolicy(BaseModel):
    """Bounded correction policy, included in every correction fingerprint."""

    schema_version: int = 1
    primary_ratio_tolerance: float = Field(default=0.10, gt=0, lt=1)
    primary_absolute_tolerance_ms: int = Field(default=250, ge=0)
    hard_ratio_tolerance: float = Field(default=0.20, gt=0, lt=1)
    max_tempo_delta: float = Field(default=0.12, gt=0, lt=0.5)
    max_provider_control_attempts: int = Field(default=1, ge=0, le=3)
    max_rewrite_attempts: int = Field(default=1, ge=0, le=3)
    max_total_attempts: int = Field(default=12, ge=1, le=32)
    pause_threshold_db: float = Field(default=-42.0, ge=-90, le=-10)
    pause_padding_ms: int = Field(default=40, ge=0, le=500)

    @model_validator(mode="after")
    def validate_tolerances(self) -> "DurationPolicy":
        if self.hard_ratio_tolerance < self.primary_ratio_tolerance:
            raise ValueError("Hard duration tolerance cannot be stricter than primary.")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint_inputs(self.model_dump(mode="json"))


class DurationRewriteResult(BaseModel):
    target_text: str
    meaning_preserved: bool
    required_terms_preserved: bool
    notes: list[str] = Field(default_factory=list)

    @field_validator("target_text")
    @classmethod
    def target_text_is_required(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Duration rewrite cannot be empty.")
        return cleaned


class DurationRewriter(Protocol):
    provider_name: str
    model_name: str

    def rewrite(
        self,
        segment: LocalizedSegment,
        *,
        available_duration_ms: int,
        generated_duration_ms: int,
        attempt_number: int,
    ) -> DurationRewriteResult:
        ...


class DurationAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    utterance_id: str
    strategy: DurationStrategy
    status: DurationAttemptStatus
    started_at: datetime
    completed_at: datetime | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    input_duration_ms: int = Field(ge=1)
    output_duration_ms: int | None = Field(default=None, ge=1)
    tempo_factor: float | None = Field(default=None, gt=0)
    speaking_rate: float | None = Field(default=None, gt=0)
    pause_scale: float | None = Field(default=None, gt=0)
    target_text_revision: int = Field(ge=1)
    provider: str | None = None
    model: str | None = None
    error_class: str | None = None
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


class DurationFitArtifact(BaseModel):
    schema_version: int = 1
    utterance_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    available_duration_ms: int = Field(gt=0)
    original_duration_ms: int = Field(gt=0)
    final_duration_ms: int = Field(gt=0)
    duration_error_ms: int
    duration_ratio: float = Field(gt=0)
    primary_tolerance_ms: int = Field(ge=0)
    within_primary_tolerance: bool
    within_hard_tolerance: bool
    status: DurationFitStatus
    selected_strategy: DurationStrategy
    target_text: str
    target_text_revision: int = Field(ge=1)
    voice_id: str
    provider: str
    model: str
    configuration_fingerprint: str
    source_speech_result_path: str
    attempts_path: str
    attempts: list[DurationAttempt]
    provider_calls: int = Field(ge=0)
    rewritten: bool = False
    needs_human_review: bool = False
    notes: list[str] = Field(default_factory=list)
    audio: ArtifactMetadata
    audio_metadata_path: str

    @model_validator(mode="after")
    def validate_timing(self) -> "DurationFitArtifact":
        if self.end_ms <= self.start_ms:
            raise ValueError("Duration-fit end_ms must follow start_ms.")
        if self.available_duration_ms != self.end_ms - self.start_ms:
            raise ValueError("Duration-fit budget must match timestamps.")
        if self.duration_error_ms != self.final_duration_ms - self.available_duration_ms:
            raise ValueError("Duration-fit error does not match measured duration.")
        expected_ratio = self.final_duration_ms / self.available_duration_ms
        if not math.isclose(self.duration_ratio, expected_ratio, abs_tol=1e-6):
            raise ValueError("Duration-fit ratio does not match measured duration.")
        if [item.attempt_number for item in self.attempts] != list(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("Duration-fit attempts are not contiguous.")
        if any(item.status == DurationAttemptStatus.RUNNING for item in self.attempts):
            raise ValueError("Completed duration fit contains a running attempt.")
        return self


class DurationMetrics(BaseModel):
    schema_version: int = 1
    configuration_fingerprint: str
    utterance_count: int = Field(ge=1)
    accepted_count: int = Field(ge=0)
    corrected_count: int = Field(ge=0)
    review_required_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    within_primary_count: int = Field(ge=0)
    within_hard_count: int = Field(ge=0)
    within_primary_percent: float = Field(ge=0, le=100)
    within_hard_percent: float = Field(ge=0, le=100)
    rewrite_count: int = Field(ge=0)
    correction_attempt_count: int = Field(ge=1)
    correction_provider_calls: int = Field(ge=0)
    maximum_absolute_error_ms: int = Field(ge=0)
    maximum_absolute_error_ratio: float = Field(ge=0)
    # Start times are pinned to immutable source offsets, so start drift cannot
    # accumulate; that is a structural property, not something measured here.
    # The measurable timing risk is corrected audio running past the next
    # utterance's cue, which render refuses outright. Measuring it here reports
    # the problem before an expensive render fails.
    next_start_overrun_count: int = Field(default=0, ge=0)
    maximum_next_start_overrun_ms: int = Field(default=0, ge=0)
    automated_timing_gate_passed: bool
    human_review_required_count: int = Field(ge=0)


class AudioDurationTransformer(Protocol):
    name: str

    def trim_artificial_pauses(
        self,
        source: Path,
        destination: Path,
        *,
        threshold_db: float,
        padding_ms: int,
    ) -> bool:
        ...

    def time_stretch(
        self,
        source: Path,
        destination: Path,
        *,
        tempo_factor: float,
    ) -> None:
        ...


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ToolResolver = Callable[[str], str | None]
ControlledSynthesizer = Callable[[Path, float, float, int], Any]
RewrittenSynthesizer = Callable[[str, int, Path, int], Any]


class WavFFmpegDurationTransformer:
    """Lossless edge-silence trim plus pitch-preserving FFmpeg `atempo`."""

    name = "wav-trim+ffmpeg-atempo"

    def __init__(
        self,
        *,
        runner: CommandRunner = subprocess.run,
        resolver: ToolResolver = shutil.which,
    ) -> None:
        self._runner = runner
        self._resolver = resolver

    def trim_artificial_pauses(
        self,
        source: Path,
        destination: Path,
        *,
        threshold_db: float,
        padding_ms: int,
    ) -> bool:
        try:
            with wave.open(str(source), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                frame_rate = handle.getframerate()
                frame_count = handle.getnframes()
                compression = handle.getcomptype()
                frames = handle.readframes(frame_count)
        except (OSError, wave.Error) as error:
            raise DurationCorrectionError(
                f"Unable to decode WAV for pause trimming: {source}"
            ) from error
        if compression != "NONE" or sample_width not in {1, 2, 3, 4}:
            raise DurationCorrectionError(
                "Pause trimming requires uncompressed 8/16/24/32-bit PCM WAV."
            )
        bytes_per_frame = channels * sample_width
        peak_value = (1 << (sample_width * 8 - 1)) - 1
        threshold = peak_value * (10 ** (threshold_db / 20))
        first = None
        last = None
        for index in range(frame_count):
            offset = index * bytes_per_frame
            frame = frames[offset : offset + bytes_per_frame]
            if _frame_peak(frame, sample_width, channels) > threshold:
                if first is None:
                    first = index
                last = index
        # All-silence provider output is invalid content, but trimming it to a
        # zero-frame WAV would hide that problem. Leave it unchanged and let
        # the unresolved timing/quality review expose it.
        if first is None or last is None:
            return False
        padding_frames = round(frame_rate * padding_ms / 1000)
        start_frame = max(0, first - padding_frames)
        end_frame = min(frame_count, last + 1 + padding_frames)
        if start_frame == 0 and end_frame == frame_count:
            return False
        trimmed = frames[
            start_frame * bytes_per_frame : end_frame * bytes_per_frame
        ]
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(destination), "wb") as handle:
                handle.setnchannels(channels)
                handle.setsampwidth(sample_width)
                handle.setframerate(frame_rate)
                handle.writeframes(trimmed)
        except (OSError, wave.Error) as error:
            raise DurationCorrectionError(
                f"Unable to write pause-trimmed WAV: {destination}"
            ) from error
        return True

    def time_stretch(
        self,
        source: Path,
        destination: Path,
        *,
        tempo_factor: float,
    ) -> None:
        ffmpeg = self._resolver("ffmpeg")
        if not ffmpeg:
            raise DurationCorrectionError(
                "FFmpeg is required for pitch-preserving duration correction."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter:a",
            f"atempo={tempo_factor:.8f}",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        result = self._runner(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "no error output"
            raise DurationCorrectionError(
                f"FFmpeg duration correction failed: {detail}"
            )


class _Candidate(BaseModel):
    path: Path
    duration_ms: int = Field(gt=0)
    target_text: str
    target_text_revision: int = Field(ge=1)
    strategy: DurationStrategy
    audio: ArtifactMetadata
    audio_metadata_path: Path
    rewritten: bool = False


class DurationCorrector:
    def __init__(
        self,
        *,
        policy: DurationPolicy | None = None,
        transformer: AudioDurationTransformer | None = None,
        rewriter: DurationRewriter | None = None,
    ) -> None:
        self.policy = policy or DurationPolicy()
        self.transformer = transformer or WavFFmpegDurationTransformer()
        self.rewriter = rewriter

    @property
    def configuration_fingerprint(self) -> str:
        return fingerprint_inputs(
            {
                "policy": self.policy.model_dump(mode="json"),
                "transformer": self.transformer.name,
                "rewriter": _rewriter_descriptor(self.rewriter),
            }
        )

    def fit(
        self,
        *,
        segment: LocalizedSegment,
        speech_result_path: Path,
        raw_audio: ArtifactMetadata,
        raw_audio_metadata_path: Path,
        run_directory: Path,
        voice_id: str,
        provider: str,
        model: str,
        controlled_synthesizer: ControlledSynthesizer | None = None,
        rewritten_synthesizer: RewrittenSynthesizer | None = None,
    ) -> tuple[DurationFitArtifact, Path, Path]:
        run_directory = run_directory.resolve()
        raw_audio_path = (run_directory / raw_audio.path).resolve()
        raw_info = wav_info(raw_audio_path)
        inputs = _fit_inputs(
            segment=segment,
            speech_result_path=speech_result_path,
            raw_audio=raw_audio,
            voice_id=voice_id,
            provider=provider,
            model=model,
            policy=self.policy,
            transformer_name=self.transformer.name,
            rewriter=self.rewriter,
        )
        fingerprint = fingerprint_inputs(inputs)
        directory = (
            run_directory
            / "speech"
            / "duration"
            / f"u-{fingerprint_inputs({'utterance_id': segment.segment_id})[:16]}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"fit-{fingerprint[:16]}"
        reusable = _find_reusable_fit(
            directory=directory,
            stem=stem,
            expected_inputs=inputs,
            segment=segment,
            raw_duration_ms=raw_info["duration_ms"],
            voice_id=voice_id,
            root=run_directory,
        )
        if reusable is not None:
            return reusable

        attempts_path = directory / f"{stem}.attempts.json"
        attempts = _load_attempts(attempts_path)
        _close_interrupted_attempts(attempts_path, attempts)
        attempts = _load_attempts(attempts_path)
        notes: list[str] = []
        candidate = _Candidate(
            path=raw_audio_path,
            duration_ms=raw_info["duration_ms"],
            target_text=segment.target_text,
            target_text_revision=segment.target_text_revision,
            strategy=DurationStrategy.ACCEPT,
            audio=raw_audio,
            audio_metadata_path=raw_audio_metadata_path,
        )

        if _correction_satisfied(
            candidate.duration_ms, segment.duration_budget_ms, self.policy
        ):
            _record_noop_attempt(
                attempts_path,
                attempts,
                strategy=DurationStrategy.ACCEPT,
                candidate=candidate,
                utterance_id=segment.segment_id,
                notes=["Generated audio is already within primary tolerance."],
            )
            attempts = _load_attempts(attempts_path)
            return self._complete_fit(
                segment=segment,
                speech_result_path=speech_result_path,
                candidate=candidate,
                original_duration_ms=raw_info["duration_ms"],
                voice_id=voice_id,
                provider=provider,
                model=model,
                inputs=inputs,
                directory=directory,
                stem=stem,
                attempts_path=attempts_path,
                attempts=attempts,
                notes=notes,
                run_directory=run_directory,
            )

        if controlled_synthesizer is None:
            notes.append("Speech provider does not expose conservative rate/pause controls.")
        else:
            for control_number in range(1, self.policy.max_provider_control_attempts + 1):
                if len(attempts) >= self.policy.max_total_attempts:
                    break
                ratio = candidate.duration_ms / segment.duration_budget_ms
                speaking_rate = min(
                    1.10,
                    max(0.90, ratio),
                )
                pause_scale = 0.75 if ratio > 1 else 1.0
                produced = self._audio_attempt(
                    attempts_path=attempts_path,
                    attempts=attempts,
                    strategy=DurationStrategy.PROVIDER_CONTROLS,
                    candidate=candidate,
                    segment=segment,
                    inputs=inputs,
                    directory=directory,
                    run_directory=run_directory,
                    provider=provider,
                    model=model,
                    parameters={
                        "speaking_rate": speaking_rate,
                        "pause_scale": pause_scale,
                        "control_attempt": control_number,
                    },
                    operation=lambda path, attempt_number: controlled_synthesizer(
                        path, speaking_rate, pause_scale, attempt_number
                    ),
                    speaking_rate=speaking_rate,
                    pause_scale=pause_scale,
                )
                attempts = _load_attempts(attempts_path)
                if produced is not None and _is_better(
                    produced.duration_ms,
                    candidate.duration_ms,
                    segment.duration_budget_ms,
                ):
                    candidate = produced
                if _correction_satisfied(
                    candidate.duration_ms,
                    segment.duration_budget_ms,
                    self.policy,
                ):
                    break

        if (
            not _correction_satisfied(candidate.duration_ms, segment.duration_budget_ms, self.policy)
            and candidate.duration_ms > segment.duration_budget_ms
            and len(attempts) < self.policy.max_total_attempts
        ):
            produced = self._audio_attempt(
                attempts_path=attempts_path,
                attempts=attempts,
                strategy=DurationStrategy.TRIM_ARTIFICIAL_PAUSES,
                candidate=candidate,
                segment=segment,
                inputs=inputs,
                directory=directory,
                run_directory=run_directory,
                provider=provider,
                model=model,
                parameters={
                    "threshold_db": self.policy.pause_threshold_db,
                    "padding_ms": self.policy.pause_padding_ms,
                },
                operation=lambda path, _attempt: self._trim(candidate.path, path),
            )
            attempts = _load_attempts(attempts_path)
            if produced is not None and _is_better(
                produced.duration_ms,
                candidate.duration_ms,
                segment.duration_budget_ms,
            ):
                candidate = produced

        if (
            not _correction_satisfied(candidate.duration_ms, segment.duration_budget_ms, self.policy)
            and len(attempts) < self.policy.max_total_attempts
        ):
            candidate = self._try_time_stretch(
                attempts_path=attempts_path,
                attempts=attempts,
                strategy=DurationStrategy.MILD_TIME_STRETCH,
                candidate=candidate,
                segment=segment,
                inputs=inputs,
                directory=directory,
                run_directory=run_directory,
                provider=provider,
                model=model,
            )
            attempts = _load_attempts(attempts_path)

        rewritten_candidate: _Candidate | None = None
        if not _correction_satisfied(
            candidate.duration_ms, segment.duration_budget_ms, self.policy
        ):
            if self.rewriter is None or rewritten_synthesizer is None:
                notes.append(
                    "Compact semantic rewriting is not configured; severe timing "
                    "violations remain visible for review."
                )
            else:
                for rewrite_number in range(1, self.policy.max_rewrite_attempts + 1):
                    if len(attempts) >= self.policy.max_total_attempts:
                        break
                    rewrite = self._rewrite_attempt(
                        attempts_path=attempts_path,
                        attempts=attempts,
                        segment=segment,
                        candidate=candidate,
                        rewrite_number=rewrite_number,
                    )
                    attempts = _load_attempts(attempts_path)
                    if rewrite is None:
                        continue
                    if len(attempts) >= self.policy.max_total_attempts:
                        break
                    revision = segment.target_text_revision + rewrite_number
                    produced = self._audio_attempt(
                        attempts_path=attempts_path,
                        attempts=attempts,
                        strategy=DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                        candidate=candidate.model_copy(
                            update={
                                "target_text": rewrite.target_text,
                                "target_text_revision": revision,
                                "rewritten": True,
                            }
                        ),
                        segment=segment,
                        inputs=inputs,
                        directory=directory,
                        run_directory=run_directory,
                        provider=provider,
                        model=model,
                        parameters={
                            "rewrite_attempt": rewrite_number,
                            "voice_id": voice_id,
                            "target_text_sha256": fingerprint_inputs(
                                {"target_text": rewrite.target_text}
                            ),
                        },
                        operation=lambda path, attempt_number, text=rewrite.target_text, rev=revision: rewritten_synthesizer(
                            text, rev, path, attempt_number
                        ),
                    )
                    attempts = _load_attempts(attempts_path)
                    if produced is not None and _is_better(
                        produced.duration_ms,
                        candidate.duration_ms,
                        segment.duration_budget_ms,
                    ):
                        rewritten_candidate = produced.model_copy(
                            update={
                                "rewritten": True,
                                "target_text": rewrite.target_text,
                                "target_text_revision": revision,
                            }
                        )
                        candidate = rewritten_candidate
                    if _correction_satisfied(
                        candidate.duration_ms,
                        segment.duration_budget_ms,
                        self.policy,
                    ):
                        break

        if (
            rewritten_candidate is not None
            and not _correction_satisfied(candidate.duration_ms, segment.duration_budget_ms, self.policy)
            and len(attempts) < self.policy.max_total_attempts
        ):
            candidate = self._try_time_stretch(
                attempts_path=attempts_path,
                attempts=attempts,
                strategy=DurationStrategy.FINAL_TIME_STRETCH,
                candidate=candidate,
                segment=segment,
                inputs=inputs,
                directory=directory,
                run_directory=run_directory,
                provider=provider,
                model=model,
            )
            attempts = _load_attempts(attempts_path)

        if not _within_hard(
            candidate.duration_ms, segment.duration_budget_ms, self.policy
        ) and len(attempts) < self.policy.max_total_attempts:
            _record_noop_attempt(
                attempts_path,
                attempts,
                strategy=DurationStrategy.SURFACE_UNRESOLVED,
                candidate=candidate,
                utterance_id=segment.segment_id,
                notes=[
                    "Duration remains outside the hard tolerance; it must not "
                    "be rendered without explicit correction or review."
                ],
            )
            attempts = _load_attempts(attempts_path)

        return self._complete_fit(
            segment=segment,
            speech_result_path=speech_result_path,
            candidate=candidate,
            original_duration_ms=raw_info["duration_ms"],
            voice_id=voice_id,
            provider=provider,
            model=model,
            inputs=inputs,
            directory=directory,
            stem=stem,
            attempts_path=attempts_path,
            attempts=attempts,
            notes=notes,
            run_directory=run_directory,
        )

    def _trim(self, source: Path, destination: Path) -> None:
        changed = self.transformer.trim_artificial_pauses(
            source,
            destination,
            threshold_db=self.policy.pause_threshold_db,
            padding_ms=self.policy.pause_padding_ms,
        )
        if not changed:
            raise DurationCorrectionError("No removable artificial edge pauses found.")

    def _try_time_stretch(
        self,
        *,
        attempts_path: Path,
        attempts: list[DurationAttempt],
        strategy: DurationStrategy,
        candidate: _Candidate,
        segment: LocalizedSegment,
        inputs: dict[str, Any],
        directory: Path,
        run_directory: Path,
        provider: str,
        model: str,
    ) -> _Candidate:
        desired = candidate.duration_ms / segment.duration_budget_ms
        tempo_factor = min(
            1 + self.policy.max_tempo_delta,
            max(1 - self.policy.max_tempo_delta, desired),
        )
        if math.isclose(tempo_factor, 1.0, abs_tol=0.001):
            return candidate
        produced = self._audio_attempt(
            attempts_path=attempts_path,
            attempts=attempts,
            strategy=strategy,
            candidate=candidate,
            segment=segment,
            inputs=inputs,
            directory=directory,
            run_directory=run_directory,
            provider=provider,
            model=model,
            parameters={"tempo_factor": tempo_factor},
            operation=lambda path, _attempt: self.transformer.time_stretch(
                candidate.path,
                path,
                tempo_factor=tempo_factor,
            ),
            tempo_factor=tempo_factor,
        )
        if produced is not None and _is_better(
            produced.duration_ms,
            candidate.duration_ms,
            segment.duration_budget_ms,
        ):
            return produced
        return candidate

    def _audio_attempt(
        self,
        *,
        attempts_path: Path,
        attempts: list[DurationAttempt],
        strategy: DurationStrategy,
        candidate: _Candidate,
        segment: LocalizedSegment,
        inputs: dict[str, Any],
        directory: Path,
        run_directory: Path,
        provider: str,
        model: str,
        parameters: dict[str, Any],
        operation: Callable[[Path, int], Any],
        tempo_factor: float | None = None,
        speaking_rate: float | None = None,
        pause_scale: float | None = None,
    ) -> _Candidate | None:
        attempt_number = len(attempts) + 1
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        running = DurationAttempt(
            attempt_number=attempt_number,
            utterance_id=segment.segment_id,
            strategy=strategy,
            status=DurationAttemptStatus.RUNNING,
            started_at=started_at,
            input_duration_ms=candidate.duration_ms,
            tempo_factor=tempo_factor,
            speaking_rate=speaking_rate,
            pause_scale=pause_scale,
            target_text_revision=candidate.target_text_revision,
            provider=(
                provider
                if strategy
                in {
                    DurationStrategy.PROVIDER_CONTROLS,
                    DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                }
                else self.transformer.name
            ),
            model=(
                model
                if strategy
                in {
                    DurationStrategy.PROVIDER_CONTROLS,
                    DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                }
                else None
            ),
        )
        _append_attempt(attempts_path, attempts, running)
        stem = f"candidate-a{attempt_number:02d}-{strategy.value}"
        output_path = directory / f"{stem}.wav"
        temporary_path = directory / f".{stem}.tmp.wav"
        if temporary_path.exists():
            temporary_path.unlink()
        try:
            operation(temporary_path, attempt_number)
            if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
                raise DurationCorrectionError(
                    f"{strategy.value} did not create non-empty audio."
                )
            _fsync_file(temporary_path)
            wav_info(temporary_path)
            os.replace(temporary_path, output_path)
            info = wav_info(output_path)
            candidate_inputs = {
                "fit": inputs,
                "strategy": strategy.value,
                "attempt_number": attempt_number,
                "input_audio_sha256": sha256_file(candidate.path),
                "parameters": parameters,
            }
            audio_metadata_path = output_path.with_name(
                output_path.name.removesuffix(".wav") + ".wav.meta.json"
            )
            audio_metadata = completed_artifact_metadata(
                artifact_id=f"{segment.segment_id}_duration_a{attempt_number:02d}",
                kind="duration_corrected_audio",
                path=output_path,
                root=run_directory,
                inputs=candidate_inputs,
                provider=(provider if strategy in {
                    DurationStrategy.PROVIDER_CONTROLS,
                    DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                } else self.transformer.name),
                model=(model if strategy in {
                    DurationStrategy.PROVIDER_CONTROLS,
                    DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                } else None),
                configuration={"strategy": strategy.value, **parameters},
            )
            write_artifact_metadata(audio_metadata_path, audio_metadata)
            completed = running.model_copy(
                update={
                    "status": DurationAttemptStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "latency_seconds": time.monotonic() - started,
                    "output_duration_ms": info["duration_ms"],
                }
            )
            _replace_attempt(attempts_path, attempts, completed)
            return _Candidate(
                path=output_path,
                duration_ms=info["duration_ms"],
                target_text=candidate.target_text,
                target_text_revision=candidate.target_text_revision,
                strategy=strategy,
                audio=audio_metadata,
                audio_metadata_path=audio_metadata_path,
                rewritten=candidate.rewritten,
            )
        except Exception as error:  # a failed optional tactic must be visible
            if temporary_path.exists():
                temporary_path.unlink()
            failed = running.model_copy(
                update={
                    "status": DurationAttemptStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "latency_seconds": time.monotonic() - started,
                    "error_class": type(error).__name__,
                    "error": redact_sensitive_text(str(error)),
                }
            )
            _replace_attempt(attempts_path, attempts, failed)
            return None

    def _rewrite_attempt(
        self,
        *,
        attempts_path: Path,
        attempts: list[DurationAttempt],
        segment: LocalizedSegment,
        candidate: _Candidate,
        rewrite_number: int,
    ) -> DurationRewriteResult | None:
        assert self.rewriter is not None
        attempt_number = len(attempts) + 1
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        running = DurationAttempt(
            attempt_number=attempt_number,
            utterance_id=segment.segment_id,
            strategy=DurationStrategy.COMPACT_REWRITE,
            status=DurationAttemptStatus.RUNNING,
            started_at=started_at,
            input_duration_ms=candidate.duration_ms,
            target_text_revision=segment.target_text_revision + rewrite_number,
            provider=self.rewriter.provider_name,
            model=self.rewriter.model_name,
        )
        _append_attempt(attempts_path, attempts, running)
        try:
            raw = self.rewriter.rewrite(
                segment,
                available_duration_ms=segment.duration_budget_ms,
                generated_duration_ms=candidate.duration_ms,
                attempt_number=rewrite_number,
            )
            result = (
                raw
                if isinstance(raw, DurationRewriteResult)
                else DurationRewriteResult.model_validate(raw)
            )
            _validate_rewrite(segment, result)
            completed = running.model_copy(
                update={
                    "status": DurationAttemptStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "latency_seconds": time.monotonic() - started,
                    "notes": result.notes,
                }
            )
            _replace_attempt(attempts_path, attempts, completed)
            return result
        except Exception as error:
            failed = running.model_copy(
                update={
                    "status": DurationAttemptStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "latency_seconds": time.monotonic() - started,
                    "error_class": type(error).__name__,
                    "error": redact_sensitive_text(str(error)),
                }
            )
            _replace_attempt(attempts_path, attempts, failed)
            return None

    def _complete_fit(
        self,
        *,
        segment: LocalizedSegment,
        speech_result_path: Path,
        candidate: _Candidate,
        original_duration_ms: int,
        voice_id: str,
        provider: str,
        model: str,
        inputs: dict[str, Any],
        directory: Path,
        stem: str,
        attempts_path: Path,
        attempts: list[DurationAttempt],
        notes: list[str],
        run_directory: Path,
    ) -> tuple[DurationFitArtifact, Path, Path]:
        within_primary = _within_primary(
            candidate.duration_ms, segment.duration_budget_ms, self.policy
        )
        within_hard = _within_hard(
            candidate.duration_ms, segment.duration_budget_ms, self.policy
        )
        if not within_hard:
            status = DurationFitStatus.UNRESOLVED
        elif candidate.rewritten or not within_primary:
            status = DurationFitStatus.REVIEW_REQUIRED
        elif candidate.strategy == DurationStrategy.ACCEPT:
            status = DurationFitStatus.ACCEPTED
        else:
            status = DurationFitStatus.CORRECTED
        error_ms = candidate.duration_ms - segment.duration_budget_ms
        artifact = DurationFitArtifact(
            utterance_id=segment.segment_id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            available_duration_ms=segment.duration_budget_ms,
            original_duration_ms=original_duration_ms,
            final_duration_ms=candidate.duration_ms,
            duration_error_ms=error_ms,
            duration_ratio=round(
                candidate.duration_ms / segment.duration_budget_ms, 8
            ),
            primary_tolerance_ms=_primary_tolerance_ms(
                segment.duration_budget_ms, self.policy
            ),
            within_primary_tolerance=within_primary,
            within_hard_tolerance=within_hard,
            status=status,
            selected_strategy=candidate.strategy,
            target_text=candidate.target_text,
            target_text_revision=candidate.target_text_revision,
            voice_id=voice_id,
            provider=provider,
            model=model,
            configuration_fingerprint=self.configuration_fingerprint,
            source_speech_result_path=relative_artifact_path(
                speech_result_path, run_directory
            ),
            attempts_path=relative_artifact_path(attempts_path, run_directory),
            attempts=attempts,
            provider_calls=sum(
                item.strategy
                in {
                    DurationStrategy.PROVIDER_CONTROLS,
                    DurationStrategy.REGENERATE_ASSIGNED_VOICE,
                }
                for item in attempts
            ),
            rewritten=candidate.rewritten,
            needs_human_review=(candidate.rewritten or not within_primary),
            notes=notes,
            audio=candidate.audio,
            audio_metadata_path=relative_artifact_path(
                candidate.audio_metadata_path, run_directory
            ),
        )
        revision = _next_revision(directory, stem)
        label = f"r{revision:04d}"
        result_path = directory / f"{stem}-{label}.result.json"
        metadata_path = directory / f"{stem}-{label}.result.meta.json"
        _write_json(result_path, artifact.model_dump(mode="json"))
        metadata = completed_artifact_metadata(
            artifact_id=f"{segment.segment_id}_duration_fit_{label}",
            kind="duration_fit_result",
            path=result_path,
            root=run_directory,
            inputs=inputs,
            provider=provider,
            model=model,
            configuration={
                "configuration_fingerprint": self.configuration_fingerprint,
                "selected_strategy": candidate.strategy.value,
            },
        )
        write_artifact_metadata(metadata_path, metadata)
        return artifact, result_path, metadata_path


def build_duration_metrics(
    artifacts: Sequence[DurationFitArtifact],
    *,
    configuration_fingerprint: str,
) -> DurationMetrics:
    if not artifacts:
        raise DurationCorrectionError("Duration metrics require at least one utterance.")
    count = len(artifacts)
    primary = sum(item.within_primary_tolerance for item in artifacts)
    hard = sum(item.within_hard_tolerance for item in artifacts)
    unresolved = sum(item.status == DurationFitStatus.UNRESOLVED for item in artifacts)
    overruns = _next_start_overruns(artifacts)
    return DurationMetrics(
        configuration_fingerprint=configuration_fingerprint,
        utterance_count=count,
        accepted_count=sum(item.status == DurationFitStatus.ACCEPTED for item in artifacts),
        corrected_count=sum(item.status == DurationFitStatus.CORRECTED for item in artifacts),
        review_required_count=sum(
            item.status == DurationFitStatus.REVIEW_REQUIRED for item in artifacts
        ),
        unresolved_count=unresolved,
        within_primary_count=primary,
        within_hard_count=hard,
        within_primary_percent=round(primary / count * 100, 3),
        within_hard_percent=round(hard / count * 100, 3),
        rewrite_count=sum(item.rewritten for item in artifacts),
        correction_attempt_count=sum(len(item.attempts) for item in artifacts),
        correction_provider_calls=sum(item.provider_calls for item in artifacts),
        maximum_absolute_error_ms=max(abs(item.duration_error_ms) for item in artifacts),
        maximum_absolute_error_ratio=max(
            abs(item.duration_ratio - 1) for item in artifacts
        ),
        next_start_overrun_count=len(overruns),
        maximum_next_start_overrun_ms=max(overruns, default=0),
        automated_timing_gate_passed=(
            primary / count >= 0.90
            and hard / count >= 0.98
            and unresolved == 0
            and not overruns
        ),
        human_review_required_count=sum(item.needs_human_review for item in artifacts),
    )


def _next_start_overruns(
    artifacts: Sequence[DurationFitArtifact],
) -> list[int]:
    """Measure how far each utterance's audio runs past the next one's cue.

    An utterance can sit inside the hard tolerance and still collide with its
    neighbour, because tolerance is measured against its own window rather than
    the gap to the next cue.
    """
    ordered = sorted(artifacts, key=lambda item: item.start_ms)
    overruns: list[int] = []
    for current, following in zip(ordered, ordered[1:]):
        overrun = (
            current.start_ms + current.final_duration_ms
        ) - following.start_ms
        if overrun > 0:
            overruns.append(overrun)
    return overruns


def load_duration_fit_artifact(path: Path) -> DurationFitArtifact:
    try:
        return DurationFitArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise DurationCorrectionError(f"Duration fit artifact is corrupt: {path}") from error


def wav_info(path: Path) -> dict[str, int]:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise OSError("file is missing or empty")
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            frame_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
    except (OSError, wave.Error) as error:
        raise DurationCorrectionError(f"Unable to decode duration WAV: {path}") from error
    if frames <= 0 or frame_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise DurationCorrectionError(f"Duration WAV has invalid parameters: {path}")
    return {
        "duration_ms": max(1, int(round(frames / frame_rate * 1000))),
        "sample_rate_hz": frame_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }


def _fit_inputs(
    *,
    segment: LocalizedSegment,
    speech_result_path: Path,
    raw_audio: ArtifactMetadata,
    voice_id: str,
    provider: str,
    model: str,
    policy: DurationPolicy,
    transformer_name: str,
    rewriter: DurationRewriter | None,
) -> dict[str, Any]:
    return {
        "utterance": segment.model_dump(mode="json"),
        "speech_result_sha256": sha256_file(speech_result_path),
        "raw_audio_sha256": raw_audio.output_sha256,
        "voice_id": voice_id,
        "provider": provider,
        "model": model,
        "policy": policy.model_dump(mode="json"),
        "transformer": transformer_name,
        "rewriter": _rewriter_descriptor(rewriter),
    }


def _rewriter_descriptor(rewriter: DurationRewriter | None) -> dict[str, str] | None:
    if rewriter is None:
        return None
    return {
        "provider": rewriter.provider_name,
        "model": rewriter.model_name,
    }


def _find_reusable_fit(
    *,
    directory: Path,
    stem: str,
    expected_inputs: dict[str, Any],
    segment: LocalizedSegment,
    raw_duration_ms: int,
    voice_id: str,
    root: Path,
) -> tuple[DurationFitArtifact, Path, Path] | None:
    for metadata_path in sorted(
        directory.glob(f"{stem}-r*.result.meta.json"), reverse=True
    ):
        result_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".json"
        )
        try:
            metadata = ArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            if not verify_artifact(
                metadata, expected_inputs=expected_inputs, root=root
            ).valid:
                continue
            artifact = load_duration_fit_artifact(result_path)
            if (
                artifact.utterance_id != segment.segment_id
                or artifact.start_ms != segment.start_ms
                or artifact.end_ms != segment.end_ms
                or artifact.available_duration_ms != segment.duration_budget_ms
                or artifact.original_duration_ms != raw_duration_ms
                or artifact.voice_id != voice_id
            ):
                continue
            audio_metadata_path = root / artifact.audio_metadata_path
            audio_metadata = ArtifactMetadata.model_validate_json(
                audio_metadata_path.read_text(encoding="utf-8")
            )
            if audio_metadata != artifact.audio:
                continue
            audio_path = root / artifact.audio.path
            if not _verify_embedded_artifact(artifact.audio, root=root):
                continue
            if wav_info(audio_path)["duration_ms"] != artifact.final_duration_ms:
                continue
            attempts = _load_attempts(root / artifact.attempts_path)
            if attempts != artifact.attempts:
                continue
            return artifact, result_path, metadata_path
        except (OSError, ValueError, ValidationError, DurationCorrectionError):
            continue
    return None


def _verify_embedded_artifact(metadata: ArtifactMetadata, *, root: Path) -> bool:
    path = root / metadata.path
    return (
        metadata.status.value == "completed"
        and path.is_file()
        and path.stat().st_size == metadata.size_bytes
        and sha256_file(path) == metadata.output_sha256
    )


def _validate_rewrite(
    segment: LocalizedSegment, result: DurationRewriteResult
) -> None:
    if not result.meaning_preserved:
        raise DurationCorrectionError("Duration rewriter did not preserve meaning.")
    if not result.required_terms_preserved:
        raise DurationCorrectionError("Duration rewriter did not preserve required terms.")
    folded = result.target_text.casefold()
    missing = [term for term in segment.glossary_terms if term.casefold() not in folded]
    if missing:
        raise DurationCorrectionError(
            "Duration rewrite omitted required glossary terms: " + ", ".join(missing)
        )


def _primary_tolerance_ms(budget_ms: int, policy: DurationPolicy) -> int:
    return max(
        policy.primary_absolute_tolerance_ms,
        round(budget_ms * policy.primary_ratio_tolerance),
    )


def _within_primary(duration_ms: int, budget_ms: int, policy: DurationPolicy) -> bool:
    return abs(duration_ms - budget_ms) <= _primary_tolerance_ms(budget_ms, policy)


def _within_hard(duration_ms: int, budget_ms: int, policy: DurationPolicy) -> bool:
    return abs(duration_ms - budget_ms) / budget_ms <= policy.hard_ratio_tolerance


def _correction_satisfied(
    duration_ms: int, budget_ms: int, policy: DurationPolicy
) -> bool:
    return _within_primary(duration_ms, budget_ms, policy) and _within_hard(
        duration_ms, budget_ms, policy
    )


def _is_better(new_ms: int, old_ms: int, budget_ms: int) -> bool:
    return abs(new_ms - budget_ms) < abs(old_ms - budget_ms)


def _record_noop_attempt(
    path: Path,
    previous: list[DurationAttempt],
    *,
    strategy: DurationStrategy,
    candidate: _Candidate,
    utterance_id: str,
    notes: list[str],
) -> None:
    now = datetime.now(timezone.utc)
    _append_attempt(
        path,
        previous,
        DurationAttempt(
            attempt_number=len(previous) + 1,
            utterance_id=utterance_id,
            strategy=strategy,
            status=DurationAttemptStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            latency_seconds=0,
            input_duration_ms=candidate.duration_ms,
            output_duration_ms=candidate.duration_ms,
            target_text_revision=candidate.target_text_revision,
            notes=notes,
        ),
    )


def _load_attempts(path: Path) -> list[DurationAttempt]:
    if not path.exists():
        return []
    try:
        attempts = [
            DurationAttempt.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise DurationCorrectionError(
            f"Duration attempt history is corrupt: {path}"
        ) from error
    if [item.attempt_number for item in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise DurationCorrectionError(
            f"Duration attempt history is not contiguous: {path}"
        )
    return attempts


def _append_attempt(
    path: Path, previous: list[DurationAttempt], attempt: DurationAttempt
) -> None:
    if attempt.attempt_number != len(previous) + 1:
        raise DurationCorrectionError("Duration attempt number is not contiguous.")
    _write_json(
        path,
        [item.model_dump(mode="json") for item in [*previous, attempt]],
    )


def _replace_attempt(
    path: Path, previous: list[DurationAttempt], attempt: DurationAttempt
) -> None:
    if attempt.attempt_number != len(previous) + 1:
        raise DurationCorrectionError("Duration attempt replacement is not contiguous.")
    _write_json(
        path,
        [item.model_dump(mode="json") for item in [*previous, attempt]],
    )


def _close_interrupted_attempts(
    path: Path, attempts: list[DurationAttempt]
) -> None:
    if not attempts or attempts[-1].status != DurationAttemptStatus.RUNNING:
        return
    interrupted = attempts[-1].model_copy(
        update={
            "status": DurationAttemptStatus.FAILED,
            "completed_at": datetime.now(timezone.utc),
            "latency_seconds": 0,
            "error_class": "interrupted",
            "error": "Process ended before the duration attempt completed.",
        }
    )
    _write_json(
        path,
        [
            item.model_dump(mode="json")
            for item in [*attempts[:-1], interrupted]
        ],
    )


def _next_revision(directory: Path, stem: str) -> int:
    revisions: list[int] = []
    for path in directory.glob(f"{stem}-r*.result.json"):
        raw = path.name.rsplit("-r", 1)[-1].removesuffix(".result.json")
        try:
            revision = int(raw)
        except ValueError:
            continue
        if revision > 0:
            revisions.append(revision)
    return max(revisions, default=0) + 1


def _frame_peak(frame: bytes, sample_width: int, channels: int) -> int:
    peak = 0
    for channel in range(channels):
        raw = frame[channel * sample_width : (channel + 1) * sample_width]
        if sample_width == 1:
            value = raw[0] - 128
        else:
            value = int.from_bytes(raw, byteorder="little", signed=True)
        peak = max(peak, abs(value))
    return peak


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

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from dub_mvp.artifacts import (
    ArtifactMetadata,
    completed_artifact_metadata,
    fingerprint_inputs,
    relative_artifact_path,
    sha256_file,
    verify_artifact,
    write_artifact_metadata,
)
from dub_mvp.manifest import redact_sensitive_text
from dub_mvp.synthesize import SynthesizedSegment


class RenderError(RuntimeError):
    retryable = True


class RenderValidationError(RenderError):
    """A permanent input/output contract failure, not a transient tool outage."""

    retryable = False


class UnresolvedDurationError(RenderValidationError):
    """Synthesis left timing outside the hard rendering boundary."""


class CompositionMode(str, Enum):
    CLEAN_REPLACEMENT = "clean_replacement"
    DUCK_ORIGINAL = "duck_original"


class RenderPolicy(BaseModel):
    schema_version: int = 1
    composition_mode: CompositionMode = CompositionMode.CLEAN_REPLACEMENT
    sample_rate_hz: int = Field(default=48_000, gt=0)
    channels: int = Field(default=2, ge=1, le=2)
    integrated_loudness_lufs: float = Field(default=-16.0, ge=-30, le=-8)
    loudness_range_lu: float = Field(default=11.0, ge=1, le=20)
    true_peak_dbfs: float = Field(default=-1.5, ge=-6, le=-0.1)
    limiter_linear_peak: float = Field(default=0.95, ge=0.1, le=1)
    edge_fade_ms: int = Field(default=8, ge=0, le=100)
    duck_volume: float = Field(default=0.18, ge=0, le=1)
    output_duration_tolerance_ms: int = Field(default=150, ge=0, le=1000)
    maximum_decoded_peak_dbfs: float = Field(default=-0.1, ge=-6, le=0)

    @property
    def configuration_fingerprint(self) -> str:
        return fingerprint_inputs(self.model_dump(mode="json"))


class AlignmentSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    tts_audio_path: str
    tts_duration_ms: int
    tempo_ratio: float
    fade_in_ms: int = Field(default=0, ge=0)
    fade_out_ms: int = Field(default=0, ge=0)
    needs_review: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timing(self) -> "AlignmentSegment":
        if self.start_ms < 0:
            raise ValueError("Alignment start_ms cannot be negative.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Alignment end_ms must be after start_ms.")
        if self.duration_budget_ms != self.end_ms - self.start_ms:
            raise ValueError("Alignment duration budget must match timestamps.")
        if self.tts_duration_ms <= 0:
            raise ValueError("TTS duration must be positive.")
        if self.tempo_ratio <= 0:
            raise ValueError("Tempo ratio must be positive.")
        effective_ms = round(self.tts_duration_ms / self.tempo_ratio)
        if self.fade_in_ms + self.fade_out_ms >= effective_ms:
            raise ValueError("Alignment fades cannot consume the utterance.")
        return self


class RenderPlan(BaseModel):
    schema_version: int = 2
    duration_ms: int
    composition_mode: CompositionMode
    sample_rate_hz: int
    channels: int
    integrated_loudness_lufs: float
    true_peak_dbfs: float
    configuration_fingerprint: str
    segments: list[AlignmentSegment]

    @model_validator(mode="after")
    def validate_segments(self) -> "RenderPlan":
        if self.duration_ms <= 0:
            raise ValueError("Render duration must be positive.")
        if not self.segments:
            raise ValueError("Render plan must contain at least one segment.")
        previous_end = 0
        identifiers: list[str] = []
        for segment in self.segments:
            if segment.end_ms > self.duration_ms:
                raise ValueError("Alignment segment exceeds render duration.")
            if segment.start_ms < previous_end:
                raise ValueError("Alignment segments cannot overlap.")
            previous_end = segment.end_ms
            identifiers.append(segment.segment_id)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Render plan contains duplicate utterance IDs.")
        return self


class ProbedStream(BaseModel):
    codec_type: str
    codec_name: str
    duration_ms: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    frame_rate: str | None = None
    sample_rate_hz: int | None = Field(default=None, ge=0)
    channels: int | None = Field(default=None, ge=0)


class ProbedMedia(BaseModel):
    path: str
    format_name: str | None = None
    duration_ms: int = Field(gt=0)
    streams: list[ProbedStream]

    def stream(self, kind: str) -> ProbedStream | None:
        return next(
            (item for item in self.streams if item.codec_type == kind), None
        )


class RenderValidation(BaseModel):
    schema_version: int = 1
    expected_duration_ms: int = Field(gt=0)
    output_duration_ms: int = Field(gt=0)
    duration_error_ms: int
    duration_within_tolerance: bool
    audio_duration_ms: int = Field(gt=0)
    audio_sample_rate_hz: int = Field(gt=0)
    audio_channels: int = Field(gt=0)
    decoded_peak_dbfs: float | None = None
    clipping_detected: bool
    full_decode_succeeded: bool
    video_stream_copied: bool
    source_video_codec: str
    output_video_codec: str
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    source_frame_rate: str | None = None
    output_frame_rate: str | None = None
    missing_utterance_ids: list[str] = Field(default_factory=list)
    duplicate_utterance_ids: list[str] = Field(default_factory=list)
    unintended_overlap_count: int = Field(default=0, ge=0)
    # Utterances are muxed at their immutable source offsets, so start drift
    # cannot accumulate. That is a structural property of the filter graph, not
    # a measurement — this field says so rather than reporting a hard-coded 0
    # that downstream gates would read as evidence.
    start_alignment_basis: str = "structural_source_offsets"
    passed: bool


class RenderCommandStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RenderCommandAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    operation: str
    status: RenderCommandStatus
    command: list[str]
    started_at: datetime
    completed_at: datetime | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    exit_code: int | None = None
    error_class: str | None = None
    error: str | None = None


class RenderArtifactReference(BaseModel):
    path: str
    metadata_path: str


class RenderReport(BaseModel):
    schema_version: int = 1
    configuration_fingerprint: str
    composition_mode: CompositionMode
    source: ProbedMedia
    dubbed_audio: ProbedMedia
    output: ProbedMedia
    validation: RenderValidation
    artifacts: dict[str, RenderArtifactReference]
    commands_path: str
    command_attempt_count: int = Field(ge=1)
    completed_at: datetime


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ToolResolver = Callable[[str], str | None]


class RenderPipeline:
    def __init__(
        self,
        runner: CommandRunner = subprocess.run,
        resolver: ToolResolver = shutil.which,
        *,
        policy: RenderPolicy | None = None,
    ) -> None:
        self._runner = runner
        self._resolver = resolver
        self.policy = policy or RenderPolicy()

    def run(
        self,
        *,
        synthesized_segments_path: Path,
        source_segment_path: Path,
        run_directory: Path,
        duration_ms: int,
        reuse_completed: bool = True,
    ) -> tuple[RenderPlan, dict[str, str]]:
        run_directory = run_directory.resolve()
        synthesized_segments_path = synthesized_segments_path.expanduser().resolve()
        source_segment_path = source_segment_path.expanduser().resolve()
        if not source_segment_path.is_file():
            raise RenderValidationError(
                f"Source segment is missing: {source_segment_path}"
            )
        ffmpeg = self._required_tool("ffmpeg")
        ffprobe = self._required_tool("ffprobe")
        synthesized_segments = load_synthesized_segments(
            synthesized_segments_path
        )
        plan = build_render_plan(
            synthesized_segments,
            duration_ms=duration_ms,
            policy=self.policy,
        )
        # Compared against the rendered plan so a dropped or duplicated
        # utterance is measured rather than assumed absent.
        requested_utterance_ids = [
            segment.segment_id for segment in synthesized_segments
        ]
        inputs = _render_inputs(
            synthesized_segments_path=synthesized_segments_path,
            source_segment_path=source_segment_path,
            segments=synthesized_segments,
            duration_ms=duration_ms,
            policy=self.policy,
        )
        fingerprint = fingerprint_inputs(inputs)
        render_directory = run_directory / "render"
        metadata_directory = run_directory / "metadata"
        subtitles_directory = run_directory / "subtitles"
        working_directory = run_directory / "working"
        output_directory = run_directory / "outputs"
        for directory in (
            render_directory,
            metadata_directory,
            subtitles_directory,
            working_directory,
            output_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        stem = f"render-{fingerprint[:16]}"
        if reuse_completed:
            reusable = _find_reusable_render(
                render_directory=render_directory,
                stem=stem,
                expected_inputs=inputs,
                root=run_directory,
            )
            if reusable is not None:
                report, outputs = reusable
                plan_path = Path(outputs["alignment_plan"])
                return (
                    RenderPlan.model_validate_json(
                        plan_path.read_text(encoding="utf-8")
                    ),
                    outputs,
                )

        revision = _select_revision(
            render_directory=render_directory,
            stem=stem,
            expected_inputs=inputs,
            root=run_directory,
            reuse_incomplete=reuse_completed,
        )
        label = f"r{revision:04d}"
        plan_path = metadata_directory / f"alignment-plan-{fingerprint[:16]}-{label}.json"
        srt_path = subtitles_directory / f"hi-{fingerprint[:16]}-{label}.srt"
        dubbed_audio_path = working_directory / f"dubbed-audio-{fingerprint[:16]}-{label}.wav"
        video_path = output_directory / f"dubbed-video-{fingerprint[:16]}-{label}.mp4"
        report_path = render_directory / f"{stem}-{label}.report.json"
        report_metadata_path = render_directory / f"{stem}-{label}.report.meta.json"
        commands_path = render_directory / f"{stem}-{label}.commands.json"

        plan_metadata_path = _metadata_path(plan_path)
        srt_metadata_path = _metadata_path(srt_path)
        audio_metadata_path = _metadata_path(dubbed_audio_path)
        video_metadata_path = _metadata_path(video_path)
        commands_metadata_path = _metadata_path(commands_path)

        if reuse_completed:
            prior = _find_reusable_artifact(
                root=run_directory,
                name_fragment=fingerprint[:16],
                kind="alignment_plan",
                expected_inputs=_artifact_inputs(inputs, "alignment_plan"),
            )
            if prior is not None:
                plan_path, plan_metadata_path = prior
            prior = _find_reusable_artifact(
                root=run_directory,
                name_fragment=fingerprint[:16],
                kind="hindi_subtitles",
                expected_inputs=_artifact_inputs(inputs, "hindi_subtitles"),
            )
            if prior is not None:
                srt_path, srt_metadata_path = prior
            prior = _find_reusable_artifact(
                root=run_directory,
                name_fragment=fingerprint[:16],
                kind="dubbed_audio",
                expected_inputs=_artifact_inputs(inputs, "dubbed_audio"),
            )
            if prior is not None:
                dubbed_audio_path, audio_metadata_path = prior
            prior = _find_reusable_artifact(
                root=run_directory,
                name_fragment=fingerprint[:16],
                kind="dubbed_video",
                expected_inputs=_artifact_inputs(inputs, "dubbed_video"),
            )
            if prior is not None:
                video_path, video_metadata_path = prior

        _close_interrupted_commands(commands_path)
        _ensure_text_artifact(
            path=plan_path,
            metadata_path=plan_metadata_path,
            payload=json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
            kind="alignment_plan",
            expected_inputs=_artifact_inputs(inputs, "alignment_plan"),
            root=run_directory,
            provider="internal",
            model=None,
        )
        _ensure_text_artifact(
            path=srt_path,
            metadata_path=srt_metadata_path,
            payload=build_srt(synthesized_segments),
            kind="hindi_subtitles",
            expected_inputs=_artifact_inputs(inputs, "hindi_subtitles"),
            root=run_directory,
            provider="internal",
            model=None,
        )

        source_probe = self._probe(
            ffprobe,
            source_segment_path,
            operation="probe source segment",
            commands_path=commands_path,
            root=run_directory,
        )
        _validate_source(source_probe, require_audio=(
            self.policy.composition_mode == CompositionMode.DUCK_ORIGINAL
        ))

        if not _artifact_reusable(
            path=dubbed_audio_path,
            metadata_path=audio_metadata_path,
            expected_inputs=_artifact_inputs(inputs, "dubbed_audio"),
            root=run_directory,
        ):
            temporary_audio = dubbed_audio_path.with_name(
                f".{dubbed_audio_path.name}.tmp.wav"
            )
            if temporary_audio.exists():
                temporary_audio.unlink()
            audio_command = _dubbed_audio_command(
                ffmpeg,
                plan,
                temporary_audio,
                policy=self.policy,
                source_segment_path=source_segment_path,
            )
            self._execute_recorded(
                audio_command,
                operation="dubbed audio assembly",
                commands_path=commands_path,
                root=run_directory,
            )
            _require_nonempty(temporary_audio, "dubbed audio assembly")
            _fsync_file(temporary_audio)
            os.replace(temporary_audio, dubbed_audio_path)
            audio_probe = self._probe(
                ffprobe,
                dubbed_audio_path,
                operation="probe dubbed audio",
                commands_path=commands_path,
                root=run_directory,
            )
            _validate_audio_probe(audio_probe, self.policy, duration_ms)
            write_artifact_metadata(
                audio_metadata_path,
                completed_artifact_metadata(
                    artifact_id=f"dubbed_audio_{label}",
                    kind="dubbed_audio",
                    path=dubbed_audio_path,
                    root=run_directory,
                    inputs=_artifact_inputs(inputs, "dubbed_audio"),
                    provider="ffmpeg",
                    configuration=self.policy.model_dump(mode="json"),
                ),
            )
        else:
            audio_probe = self._probe(
                ffprobe,
                dubbed_audio_path,
                operation="verify reused dubbed audio",
                commands_path=commands_path,
                root=run_directory,
            )
            _validate_audio_probe(audio_probe, self.policy, duration_ms)

        dubbed_audio_peak_dbfs = self._measure_peak(
            ffmpeg,
            dubbed_audio_path,
            commands_path=commands_path,
            root=run_directory,
        )
        if (
            dubbed_audio_peak_dbfs is not None
            and dubbed_audio_peak_dbfs > self.policy.maximum_decoded_peak_dbfs
        ):
            raise RenderValidationError(
                f"Dubbed audio peak {dubbed_audio_peak_dbfs:.2f} dBFS exceeds "
                f"{self.policy.maximum_decoded_peak_dbfs:.2f} dBFS."
            )

        if not _artifact_reusable(
            path=video_path,
            metadata_path=video_metadata_path,
            expected_inputs=_artifact_inputs(inputs, "dubbed_video"),
            root=run_directory,
        ):
            temporary_video = video_path.with_name(
                f".{video_path.name}.tmp.mp4"
            )
            if temporary_video.exists():
                temporary_video.unlink()
            self._execute_recorded(
                _mux_command(
                    ffmpeg,
                    source_segment_path=source_segment_path,
                    dubbed_audio_path=dubbed_audio_path,
                    output_path=temporary_video,
                    duration_ms=duration_ms,
                ),
                operation="dubbed video mux",
                commands_path=commands_path,
                root=run_directory,
            )
            _require_nonempty(temporary_video, "dubbed video mux")
            _fsync_file(temporary_video)
            os.replace(temporary_video, video_path)
            output_probe = self._probe(
                ffprobe,
                video_path,
                operation="probe dubbed video",
                commands_path=commands_path,
                root=run_directory,
            )
            self._decode(
                ffmpeg,
                video_path,
                commands_path=commands_path,
                root=run_directory,
            )
            output_peak_dbfs = self._measure_peak(
                ffmpeg,
                video_path,
                commands_path=commands_path,
                root=run_directory,
            )
            validation = _validate_rendered_media(
                plan=plan,
                requested_utterance_ids=requested_utterance_ids,
                source=source_probe,
                audio=audio_probe,
                output=output_probe,
                peak_dbfs=output_peak_dbfs,
                policy=self.policy,
            )
            write_artifact_metadata(
                video_metadata_path,
                completed_artifact_metadata(
                    artifact_id=f"dubbed_video_{label}",
                    kind="dubbed_video",
                    path=video_path,
                    root=run_directory,
                    inputs=_artifact_inputs(inputs, "dubbed_video"),
                    provider="ffmpeg",
                    configuration={
                        "video_codec": "copy",
                        "audio_codec": "aac",
                        "validation_passed": validation.passed,
                    },
                ),
            )
        else:
            output_probe = self._probe(
                ffprobe,
                video_path,
                operation="verify reused dubbed video",
                commands_path=commands_path,
                root=run_directory,
            )
            self._decode(
                ffmpeg,
                video_path,
                commands_path=commands_path,
                root=run_directory,
            )
            output_peak_dbfs = self._measure_peak(
                ffmpeg,
                video_path,
                commands_path=commands_path,
                root=run_directory,
            )
            validation = _validate_rendered_media(
                plan=plan,
                requested_utterance_ids=requested_utterance_ids,
                source=source_probe,
                audio=audio_probe,
                output=output_probe,
                peak_dbfs=output_peak_dbfs,
                policy=self.policy,
            )

        for path, metadata_path, kind, provider in (
            (plan_path, plan_metadata_path, "alignment_plan", "internal"),
            (srt_path, srt_metadata_path, "hindi_subtitles", "internal"),
        ):
            if not _artifact_reusable(
                path=path,
                metadata_path=metadata_path,
                expected_inputs=_artifact_inputs(inputs, kind),
                root=run_directory,
            ):
                raise RenderValidationError(f"Render artifact failed verification: {path}")

        commands = _load_commands(commands_path)
        if any(item.status == RenderCommandStatus.RUNNING for item in commands):
            raise RenderValidationError("Render command history contains running work.")
        write_artifact_metadata(
            commands_metadata_path,
            completed_artifact_metadata(
                artifact_id=f"render_commands_{label}",
                kind="render_commands",
                path=commands_path,
                root=run_directory,
                inputs=_artifact_inputs(inputs, "render_commands"),
                provider="ffmpeg",
            ),
        )
        artifact_paths = {
            "alignment_plan": (plan_path, plan_metadata_path),
            "hindi_srt": (srt_path, srt_metadata_path),
            "dubbed_audio": (dubbed_audio_path, audio_metadata_path),
            "dubbed_video": (video_path, video_metadata_path),
            "render_commands": (commands_path, commands_metadata_path),
        }
        report = RenderReport(
            configuration_fingerprint=fingerprint,
            composition_mode=self.policy.composition_mode,
            source=source_probe,
            dubbed_audio=audio_probe,
            output=output_probe,
            validation=validation,
            artifacts={
                name: RenderArtifactReference(
                    path=relative_artifact_path(path, run_directory),
                    metadata_path=relative_artifact_path(metadata, run_directory),
                )
                for name, (path, metadata) in artifact_paths.items()
            },
            commands_path=relative_artifact_path(commands_path, run_directory),
            command_attempt_count=len(commands),
            completed_at=datetime.now(timezone.utc),
        )
        _write_text_atomic(
            report_path,
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=True)
            + "\n",
        )
        write_artifact_metadata(
            report_metadata_path,
            completed_artifact_metadata(
                artifact_id=f"render_report_{label}",
                kind="render_report",
                path=report_path,
                root=run_directory,
                inputs=_artifact_inputs(inputs, "render_report"),
                provider="ffmpeg",
                configuration={
                    "configuration_fingerprint": fingerprint,
                    "validation_passed": validation.passed,
                },
            ),
        )
        outputs = {
            name: str(path.resolve())
            for name, (path, _) in artifact_paths.items()
        }
        outputs.update(
            {
                f"{name}_metadata": str(metadata.resolve())
                for name, (_, metadata) in artifact_paths.items()
            }
        )
        outputs["render_report"] = str(report_path.resolve())
        outputs["render_report_metadata"] = str(report_metadata_path.resolve())
        return plan, outputs

    def _required_tool(self, name: str) -> str:
        resolved = self._resolver(name)
        if not resolved:
            raise RenderError(f"Required render tool '{name}' was not found.")
        return resolved

    def _execute_recorded(
        self,
        command: Sequence[str],
        *,
        operation: str,
        commands_path: Path,
        root: Path,
    ) -> subprocess.CompletedProcess[str]:
        previous = _load_commands(commands_path)
        attempt_number = len(previous) + 1
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        running = RenderCommandAttempt(
            attempt_number=attempt_number,
            operation=operation,
            status=RenderCommandStatus.RUNNING,
            command=_portable_command(command, root),
            started_at=started_at,
        )
        _write_commands(commands_path, [*previous, running])
        try:
            result = self._runner(
                list(command),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as error:
            failed = running.model_copy(
                update={
                    "status": RenderCommandStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "latency_seconds": time.monotonic() - started,
                    "error_class": type(error).__name__,
                    "error": redact_sensitive_text(str(error)),
                }
            )
            _write_commands(commands_path, [*previous, failed])
            raise RenderError(
                f"Failed {operation}: {type(error).__name__}: {error}"
            ) from error
        completed = running.model_copy(
            update={
                "status": (
                    RenderCommandStatus.COMPLETED
                    if result.returncode == 0
                    else RenderCommandStatus.FAILED
                ),
                "completed_at": datetime.now(timezone.utc),
                "latency_seconds": time.monotonic() - started,
                "exit_code": result.returncode,
                "error_class": (
                    None if result.returncode == 0 else "command_failed"
                ),
                "error": (
                    None
                    if result.returncode == 0
                    else redact_sensitive_text(
                        result.stderr.strip() or "no error output"
                    )
                ),
            }
        )
        _write_commands(commands_path, [*previous, completed])
        if result.returncode != 0:
            raise RenderError(
                f"Failed {operation}: {completed.error or 'no error output'}"
            )
        return result

    def _probe(
        self,
        ffprobe: str,
        path: Path,
        *,
        operation: str,
        commands_path: Path,
        root: Path,
    ) -> ProbedMedia:
        result = self._execute_recorded(
            [
                ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            operation=operation,
            commands_path=commands_path,
            root=root,
        )
        try:
            payload = json.loads(result.stdout)
            return _parse_probe(payload, path=path, root=root)
        except (ValueError, TypeError, KeyError, ValidationError) as error:
            raise RenderValidationError(
                f"FFprobe returned invalid metadata for {path}: {error}"
            ) from error

    def _measure_peak(
        self,
        ffmpeg: str,
        audio_path: Path,
        *,
        commands_path: Path,
        root: Path,
    ) -> float | None:
        result = self._execute_recorded(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                str(audio_path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            operation="measure decoded audio peak",
            commands_path=commands_path,
            root=root,
        )
        match = re.search(r"max_volume:\s*(-?inf|[-+]?\d+(?:\.\d+)?)\s*dB", result.stderr)
        if not match or match.group(1) == "-inf":
            return None
        return float(match.group(1))

    def _decode(
        self,
        ffmpeg: str,
        video_path: Path,
        *,
        commands_path: Path,
        root: Path,
    ) -> None:
        self._execute_recorded(
            [
                ffmpeg,
                "-hide_banner",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            operation="full output decode validation",
            commands_path=commands_path,
            root=root,
        )


def load_synthesized_segments(path: Path) -> list[SynthesizedSegment]:
    if not path.is_file():
        raise RenderValidationError(f"Synthesized segments are missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [SynthesizedSegment.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise RenderValidationError(
            f"Unable to read synthesized segments: {path}"
        ) from error
    if not segments:
        raise RenderValidationError(f"Synthesized segments are empty: {path}")
    identifiers = [item.segment_id for item in segments]
    if len(identifiers) != len(set(identifiers)):
        raise RenderValidationError("Synthesized segments contain duplicate IDs.")
    return segments


def build_render_plan(
    segments: Sequence[SynthesizedSegment],
    *,
    duration_ms: int,
    review_threshold: float = 0.15,
    hard_limit: float = 0.20,
    policy: RenderPolicy | None = None,
) -> RenderPlan:
    active_policy = policy or RenderPolicy()
    aligned: list[AlignmentSegment] = []
    previous_end = 0
    for segment in segments:
        if segment.start_ms < previous_end:
            raise RenderValidationError(f"Segment overlap at {segment.segment_id}.")
        audio_path = Path(segment.tts_audio_path)
        if not audio_path.is_file():
            raise RenderValidationError(
                f"Synthesized audio is missing for {segment.segment_id}: "
                f"{audio_path}"
            )
        drift = (
            segment.tts_duration_ms - segment.duration_budget_ms
        ) / segment.duration_budget_ms
        if segment.schema_version >= 3:
            if segment.duration_status == "unresolved" or abs(drift) > hard_limit:
                raise UnresolvedDurationError(
                    f"{segment.segment_id} has an unresolved duration violation "
                    f"of {abs(drift):.1%}; synthesis must correct or explicitly "
                    "review it before render."
                )
            tempo_ratio = 1.0
            needs_review = segment.requires_timing_review
            notes = (
                ["Duration fitting marked this utterance for human review."]
                if needs_review
                else []
            )
        else:
            if abs(drift) > hard_limit:
                raise UnresolvedDurationError(
                    f"{segment.segment_id} requires {abs(drift):.1%} tempo "
                    "correction, above the 20% hard limit."
                )
            tempo_ratio = round(
                segment.tts_duration_ms / segment.duration_budget_ms, 6
            )
            needs_review = abs(drift) > review_threshold
            notes = (
                [f"Tempo correction {abs(drift):.1%} exceeds review threshold."]
                if needs_review
                else []
            )
        effective_duration_ms = max(1, round(segment.tts_duration_ms / tempo_ratio))
        fade_ms = min(
            active_policy.edge_fade_ms,
            max(0, (effective_duration_ms - 1) // 2),
        )
        aligned.append(
            AlignmentSegment(
                segment_id=segment.segment_id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                duration_budget_ms=segment.duration_budget_ms,
                tts_audio_path=segment.tts_audio_path,
                tts_duration_ms=segment.tts_duration_ms,
                tempo_ratio=tempo_ratio,
                fade_in_ms=fade_ms,
                fade_out_ms=fade_ms,
                needs_review=needs_review,
                notes=notes,
            )
        )
        previous_end = segment.end_ms
    for current, following in zip(aligned, aligned[1:]):
        effective_end_ms = current.start_ms + round(
            current.tts_duration_ms / current.tempo_ratio
        )
        if effective_end_ms > following.start_ms:
            raise UnresolvedDurationError(
                f"{current.segment_id} audio would overlap "
                f"{following.segment_id} by "
                f"{effective_end_ms - following.start_ms} ms."
            )
    final = aligned[-1]
    final_audio_end = final.start_ms + round(
        final.tts_duration_ms / final.tempo_ratio
    )
    if final_audio_end > duration_ms:
        raise UnresolvedDurationError(
            f"{final.segment_id} audio exceeds the output timeline by "
            f"{final_audio_end - duration_ms} ms."
        )
    return RenderPlan(
        duration_ms=duration_ms,
        composition_mode=active_policy.composition_mode,
        sample_rate_hz=active_policy.sample_rate_hz,
        channels=active_policy.channels,
        integrated_loudness_lufs=active_policy.integrated_loudness_lufs,
        true_peak_dbfs=active_policy.true_peak_dbfs,
        configuration_fingerprint=active_policy.configuration_fingerprint,
        segments=aligned,
    )


def build_srt(segments: Sequence[SynthesizedSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    (
                        f"{_srt_time(segment.start_ms)} --> "
                        f"{_srt_time(segment.end_ms)}"
                    ),
                    segment.target_text,
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def render_outputs_reusable(
    *,
    outputs: dict[str, str],
    synthesized_segments_path: Path,
    source_segment_path: Path,
    run_directory: Path,
    duration_ms: int,
    policy: RenderPolicy | None = None,
) -> bool:
    required = {"render_report", "render_report_metadata"}
    if not required.issubset(outputs):
        return False
    try:
        active_policy = policy or RenderPolicy()
        segments = load_synthesized_segments(synthesized_segments_path)
        inputs = _render_inputs(
            synthesized_segments_path=synthesized_segments_path,
            source_segment_path=source_segment_path,
            segments=segments,
            duration_ms=duration_ms,
            policy=active_policy,
        )
        report_path = Path(outputs["render_report"])
        metadata_path = Path(outputs["render_report_metadata"])
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if (
            (run_directory / metadata.path).resolve() != report_path.resolve()
            or not verify_artifact(
                metadata,
                expected_inputs=_artifact_inputs(inputs, "render_report"),
                root=run_directory,
            ).valid
        ):
            return False
        report = RenderReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        if (
            report.configuration_fingerprint != fingerprint_inputs(inputs)
            or not report.validation.passed
        ):
            return False
        return _report_artifacts_reusable(
            report,
            expected_inputs=inputs,
            root=run_directory,
        )
    except (OSError, ValueError, TypeError, ValidationError, RenderError):
        return False


def _dubbed_audio_command(
    ffmpeg: str,
    plan: RenderPlan,
    output_path: Path,
    *,
    policy: RenderPolicy,
    source_segment_path: Path,
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    input_offset = 0
    if policy.composition_mode == CompositionMode.DUCK_ORIGINAL:
        command.extend(["-i", str(source_segment_path)])
        input_offset = 1
    for segment in plan.segments:
        command.extend(["-i", segment.tts_audio_path])

    duration_seconds = plan.duration_ms / 1000
    channel_layout = "stereo" if policy.channels == 2 else "mono"
    filters: list[str] = []
    if policy.composition_mode == CompositionMode.DUCK_ORIGINAL:
        windows = "+".join(
            f"between(t,{item.start_ms / 1000:.3f},{item.end_ms / 1000:.3f})"
            for item in plan.segments
        )
        filters.append(
            "[0:a]"
            f"aresample={policy.sample_rate_hz},"
            f"aformat=sample_fmts=fltp:channel_layouts={channel_layout},"
            f"volume='if(gt({windows},0),{policy.duck_volume:.6f},1.0)':eval=frame,"
            f"atrim=end={duration_seconds:.3f},asetpts=N/SR/TB[bed]"
        )
    else:
        filters.append(
            f"anullsrc=r={policy.sample_rate_hz}:cl={channel_layout}:"
            f"d={duration_seconds:.3f}[bed]"
        )

    labels: list[str] = []
    for index, segment in enumerate(plan.segments):
        label = f"voice{index}"
        effective_seconds = segment.tts_duration_ms / segment.tempo_ratio / 1000
        chain = (
            f"[{index + input_offset}:a]"
            f"aresample={policy.sample_rate_hz},"
            f"aformat=sample_fmts=fltp:channel_layouts={channel_layout},"
            f"atempo={segment.tempo_ratio:.6f}"
        )
        if segment.fade_in_ms:
            chain += f",afade=t=in:st=0:d={segment.fade_in_ms / 1000:.3f}"
        if segment.fade_out_ms:
            fade_start = max(0, effective_seconds - segment.fade_out_ms / 1000)
            chain += (
                f",afade=t=out:st={fade_start:.3f}:"
                f"d={segment.fade_out_ms / 1000:.3f}"
            )
        chain += f",adelay={segment.start_ms}:all=1[{label}]"
        filters.append(chain)
        labels.append(f"[{label}]")

    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[speech]")
    else:
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:"
            "duration=longest:normalize=0:dropout_transition=0[speech]"
        )
    filters.append(
        "[bed][speech]amix=inputs=2:duration=first:normalize=0:"
        "dropout_transition=0,"
        f"loudnorm=I={policy.integrated_loudness_lufs:.2f}:"
        f"LRA={policy.loudness_range_lu:.2f}:TP={policy.true_peak_dbfs:.2f},"
        f"alimiter=limit={policy.limiter_linear_peak:.6f}:attack=5:release=50,"
        f"atrim=end={duration_seconds:.3f},asetpts=N/SR/TB[dub]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[dub]",
            "-ar",
            str(policy.sample_rate_hz),
            "-ac",
            str(policy.channels),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return command


def _mux_command(
    ffmpeg: str,
    *,
    source_segment_path: Path,
    dubbed_audio_path: Path,
    output_path: Path,
    duration_ms: int,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_segment_path),
        "-i",
        str(dubbed_audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _render_inputs(
    *,
    synthesized_segments_path: Path,
    source_segment_path: Path,
    segments: Sequence[SynthesizedSegment],
    duration_ms: int,
    policy: RenderPolicy,
) -> dict[str, Any]:
    return {
        "synthesized_segments_sha256": sha256_file(synthesized_segments_path),
        "source_segment_sha256": sha256_file(source_segment_path),
        "utterance_ids": [item.segment_id for item in segments],
        "utterance_audio_sha256": [
            sha256_file(Path(item.tts_audio_path)) for item in segments
        ],
        "duration_ms": duration_ms,
        "policy": policy.model_dump(mode="json"),
        "render_contract": "phase11_v1",
    }


def _artifact_inputs(inputs: dict[str, Any], kind: str) -> dict[str, Any]:
    return {"render": inputs, "kind": kind}


def _ensure_text_artifact(
    *,
    path: Path,
    metadata_path: Path,
    payload: str,
    kind: str,
    expected_inputs: dict[str, Any],
    root: Path,
    provider: str,
    model: str | None,
) -> None:
    if _artifact_reusable(
        path=path,
        metadata_path=metadata_path,
        expected_inputs=expected_inputs,
        root=root,
    ):
        return
    _write_text_atomic(path, payload)
    write_artifact_metadata(
        metadata_path,
        completed_artifact_metadata(
            artifact_id=path.stem,
            kind=kind,
            path=path,
            root=root,
            inputs=expected_inputs,
            provider=provider,
            model=model,
        ),
    )


def _artifact_reusable(
    *,
    path: Path,
    metadata_path: Path,
    expected_inputs: dict[str, Any],
    root: Path,
) -> bool:
    try:
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        return (
            (root / metadata.path).resolve() == path.resolve()
            and verify_artifact(
                metadata, expected_inputs=expected_inputs, root=root
            ).valid
        )
    except (OSError, ValueError, ValidationError):
        return False


def _find_reusable_artifact(
    *,
    root: Path,
    name_fragment: str,
    kind: str,
    expected_inputs: dict[str, Any],
) -> tuple[Path, Path] | None:
    for metadata_path in sorted(
        root.glob(f"**/*{name_fragment}*.meta.json"), reverse=True
    ):
        try:
            metadata = ArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            path = root / metadata.path
            if metadata.kind != kind:
                continue
            if _artifact_reusable(
                path=path,
                metadata_path=metadata_path,
                expected_inputs=expected_inputs,
                root=root,
            ):
                return path, metadata_path
        except (OSError, ValueError, ValidationError):
            continue
    return None


def _find_reusable_render(
    *,
    render_directory: Path,
    stem: str,
    expected_inputs: dict[str, Any],
    root: Path,
) -> tuple[RenderReport, dict[str, str]] | None:
    for metadata_path in sorted(
        render_directory.glob(f"{stem}-r*.report.meta.json"), reverse=True
    ):
        report_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".json"
        )
        try:
            metadata = ArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            if not verify_artifact(
                metadata,
                expected_inputs=_artifact_inputs(expected_inputs, "render_report"),
                root=root,
            ).valid:
                continue
            report = RenderReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
            if (
                report.configuration_fingerprint
                != fingerprint_inputs(expected_inputs)
                or not report.validation.passed
                or not _report_artifacts_reusable(
                    report, expected_inputs=expected_inputs, root=root
                )
            ):
                continue
            outputs = {
                name: str((root / reference.path).resolve())
                for name, reference in report.artifacts.items()
            }
            outputs.update(
                {
                    f"{name}_metadata": str(
                        (root / reference.metadata_path).resolve()
                    )
                    for name, reference in report.artifacts.items()
                }
            )
            outputs["render_report"] = str(report_path.resolve())
            outputs["render_report_metadata"] = str(metadata_path.resolve())
            return report, outputs
        except (OSError, ValueError, ValidationError):
            continue
    return None


def _report_artifacts_reusable(
    report: RenderReport,
    *,
    expected_inputs: dict[str, Any],
    root: Path,
) -> bool:
    required = {
        "alignment_plan",
        "hindi_srt",
        "dubbed_audio",
        "dubbed_video",
        "render_commands",
    }
    if not required.issubset(report.artifacts):
        return False
    kinds = {
        "alignment_plan": "alignment_plan",
        "hindi_srt": "hindi_subtitles",
        "dubbed_audio": "dubbed_audio",
        "dubbed_video": "dubbed_video",
        "render_commands": "render_commands",
    }
    return all(
        _artifact_reusable(
            path=root / reference.path,
            metadata_path=root / reference.metadata_path,
            expected_inputs=_artifact_inputs(expected_inputs, kinds[name]),
            root=root,
        )
        for name, reference in report.artifacts.items()
        if name in kinds
    )


def _select_revision(
    *,
    render_directory: Path,
    stem: str,
    expected_inputs: dict[str, Any],
    root: Path,
    reuse_incomplete: bool,
) -> int:
    revisions = _render_revisions(render_directory, stem)
    if not revisions:
        return 1
    latest = max(revisions)
    if not reuse_incomplete:
        return latest + 1
    label = f"r{latest:04d}"
    report_metadata = render_directory / f"{stem}-{label}.report.meta.json"
    if report_metadata.exists():
        return latest + 1
    # A partial revision is safe to resume only when every sidecar it already
    # published still proves its output. Invalid completed work gets a new
    # revision rather than being overwritten.
    for metadata_path in render_directory.parent.glob(f"**/*{stem[7:]}-{label}*.meta.json"):
        try:
            metadata = ArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            kind = metadata.kind
            normalized = {
                "hindi_subtitles": "hindi_subtitles",
                "alignment_plan": "alignment_plan",
                "dubbed_audio": "dubbed_audio",
                "dubbed_video": "dubbed_video",
                "render_commands": "render_commands",
            }.get(kind)
            if normalized is None or not verify_artifact(
                metadata,
                expected_inputs=_artifact_inputs(expected_inputs, normalized),
                root=root,
            ).valid:
                return latest + 1
        except (OSError, ValueError, ValidationError):
            return latest + 1
    return latest


def _render_revisions(directory: Path, stem: str) -> list[int]:
    revisions: set[int] = set()
    fragment = stem.removeprefix("render-")
    # Revision evidence is distributed across metadata/, working/, outputs/,
    # subtitles/, and render/. Losing the command/report directory must never
    # make the writer reuse r0001 and overwrite a verified media artifact.
    for path in directory.parent.glob(f"**/*{fragment}-r*"):
        match = re.search(r"-r(\d{4})", path.name)
        if match:
            revisions.add(int(match.group(1)))
    return sorted(revisions)


def _parse_probe(payload: dict[str, Any], *, path: Path, root: Path) -> ProbedMedia:
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise ValueError("FFprobe streams are missing.")
    duration_ms = _duration_ms(payload.get("format", {}).get("duration"))
    streams: list[ProbedStream] = []
    for raw in raw_streams:
        if not isinstance(raw, dict):
            continue
        codec_type = str(raw.get("codec_type", "")).strip()
        codec_name = str(raw.get("codec_name", "")).strip()
        if not codec_type or not codec_name:
            continue
        stream_duration = _duration_ms(raw.get("duration"), required=False)
        streams.append(
            ProbedStream(
                codec_type=codec_type,
                codec_name=codec_name,
                duration_ms=stream_duration,
                width=_optional_int(raw.get("width")),
                height=_optional_int(raw.get("height")),
                frame_rate=(
                    str(raw.get("avg_frame_rate"))
                    if raw.get("avg_frame_rate") is not None
                    else None
                ),
                sample_rate_hz=_optional_int(raw.get("sample_rate")),
                channels=_optional_int(raw.get("channels")),
            )
        )
    if duration_ms is None:
        candidates = [item.duration_ms for item in streams if item.duration_ms]
        if not candidates:
            raise ValueError("FFprobe duration is missing.")
        duration_ms = max(candidates)
    try:
        display_path = relative_artifact_path(path, root)
    except ValueError:
        display_path = path.name
    return ProbedMedia(
        path=display_path,
        format_name=(
            str(payload.get("format", {}).get("format_name"))
            if payload.get("format", {}).get("format_name") is not None
            else None
        ),
        duration_ms=duration_ms,
        streams=streams,
    )


def _validate_source(source: ProbedMedia, *, require_audio: bool) -> None:
    video = source.stream("video")
    if video is None or not video.width or not video.height:
        raise RenderValidationError("Source segment has no valid video stream.")
    if require_audio and source.stream("audio") is None:
        raise RenderValidationError(
            "Original-track ducking requires a source audio stream."
        )


def _validate_audio_probe(
    audio: ProbedMedia, policy: RenderPolicy, expected_duration_ms: int
) -> None:
    stream = audio.stream("audio")
    if stream is None:
        raise RenderValidationError("Dubbed WAV has no audio stream.")
    if stream.sample_rate_hz != policy.sample_rate_hz:
        raise RenderValidationError(
            f"Dubbed WAV sample rate is {stream.sample_rate_hz}, expected "
            f"{policy.sample_rate_hz}."
        )
    if stream.channels != policy.channels:
        raise RenderValidationError(
            f"Dubbed WAV has {stream.channels} channels, expected {policy.channels}."
        )
    if abs(audio.duration_ms - expected_duration_ms) > policy.output_duration_tolerance_ms:
        raise RenderValidationError(
            f"Dubbed WAV duration differs by "
            f"{abs(audio.duration_ms - expected_duration_ms)} ms."
        )


def _validate_rendered_media(
    *,
    plan: RenderPlan,
    requested_utterance_ids: Sequence[str],
    source: ProbedMedia,
    audio: ProbedMedia,
    output: ProbedMedia,
    peak_dbfs: float | None,
    policy: RenderPolicy,
) -> RenderValidation:
    source_video = source.stream("video")
    output_video = output.stream("video")
    output_audio = output.stream("audio")
    if source_video is None or output_video is None or output_audio is None:
        raise RenderValidationError(
            "Rendered output must contain one decodable video and audio stream."
        )
    duration_error = output.duration_ms - plan.duration_ms
    duration_ok = abs(duration_error) <= policy.output_duration_tolerance_ms
    copied = (
        source_video.codec_name == output_video.codec_name
        and source_video.width == output_video.width
        and source_video.height == output_video.height
        and source_video.frame_rate == output_video.frame_rate
    )
    clipping = (
        peak_dbfs is not None
        and peak_dbfs > policy.maximum_decoded_peak_dbfs
    )

    # Derive integrity from the plan actually rendered rather than asserting it.
    planned_ids = [segment.segment_id for segment in plan.segments]
    seen: set[str] = set()
    duplicate_ids = sorted(
        {item for item in planned_ids if item in seen or seen.add(item)}
    )
    missing_ids = sorted(set(requested_utterance_ids) - set(planned_ids))
    overlap_count = sum(
        1
        for current, following in zip(plan.segments, plan.segments[1:])
        if current.start_ms
        + round(current.tts_duration_ms / current.tempo_ratio)
        > following.start_ms
    )
    validation = RenderValidation(
        expected_duration_ms=plan.duration_ms,
        output_duration_ms=output.duration_ms,
        duration_error_ms=duration_error,
        duration_within_tolerance=duration_ok,
        audio_duration_ms=audio.duration_ms,
        audio_sample_rate_hz=output_audio.sample_rate_hz or 0,
        audio_channels=output_audio.channels or 0,
        decoded_peak_dbfs=peak_dbfs,
        clipping_detected=clipping,
        full_decode_succeeded=True,
        video_stream_copied=copied,
        source_video_codec=source_video.codec_name,
        output_video_codec=output_video.codec_name,
        source_width=source_video.width or 0,
        source_height=source_video.height or 0,
        output_width=output_video.width or 0,
        output_height=output_video.height or 0,
        source_frame_rate=source_video.frame_rate,
        output_frame_rate=output_video.frame_rate,
        missing_utterance_ids=missing_ids,
        duplicate_utterance_ids=duplicate_ids,
        unintended_overlap_count=overlap_count,
        passed=(
            duration_ok
            and not clipping
            and copied
            and not missing_ids
            and not duplicate_ids
            and overlap_count == 0
        ),
    )
    if validation.audio_sample_rate_hz != policy.sample_rate_hz:
        raise RenderValidationError(
            f"Output sample rate is {validation.audio_sample_rate_hz}, expected "
            f"{policy.sample_rate_hz}."
        )
    if validation.audio_channels != policy.channels:
        raise RenderValidationError(
            f"Output has {validation.audio_channels} channels, expected "
            f"{policy.channels}."
        )
    if not duration_ok:
        raise RenderValidationError(
            f"Rendered duration differs by {abs(duration_error)} ms."
        )
    if not copied:
        raise RenderValidationError(
            "Video stream codec, dimensions, or frame rate changed despite copy mode."
        )
    if clipping:
        raise RenderValidationError("Rendered audio clipping was detected.")
    return validation


def _report_outputs(report: RenderReport, root: Path) -> dict[str, str]:
    outputs = {
        name: str((root / reference.path).resolve())
        for name, reference in report.artifacts.items()
    }
    outputs.update(
        {
            f"{name}_metadata": str(
                (root / reference.metadata_path).resolve()
            )
            for name, reference in report.artifacts.items()
        }
    )
    return outputs


def _load_commands(path: Path) -> list[RenderCommandAttempt]:
    if not path.exists():
        return []
    try:
        attempts = [
            RenderCommandAttempt.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise RenderValidationError(
            f"Render command history is corrupt: {path}"
        ) from error
    if [item.attempt_number for item in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise RenderValidationError(
            f"Render command history is not contiguous: {path}"
        )
    return attempts


def _write_commands(path: Path, attempts: Sequence[RenderCommandAttempt]) -> None:
    _write_text_atomic(
        path,
        json.dumps(
            [item.model_dump(mode="json") for item in attempts],
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
    )


def _close_interrupted_commands(path: Path) -> None:
    attempts = _load_commands(path)
    if not attempts or attempts[-1].status != RenderCommandStatus.RUNNING:
        return
    attempts[-1] = attempts[-1].model_copy(
        update={
            "status": RenderCommandStatus.FAILED,
            "completed_at": datetime.now(timezone.utc),
            "latency_seconds": 0,
            "error_class": "interrupted",
            "error": "Process ended before the render command completed.",
        }
    )
    _write_commands(path, attempts)


def _portable_command(command: Sequence[str], root: Path) -> list[str]:
    root_text = str(root.resolve())
    portable = []
    for item in command:
        value = str(item)
        if value == root_text:
            value = "<run>"
        elif value.startswith(root_text + os.sep):
            value = "<run>/" + value[len(root_text + os.sep) :]
        portable.append(redact_sensitive_text(value))
    return portable


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _duration_ms(value: Any, *, required: bool = True) -> int | None:
    if value in {None, "N/A"}:
        if required:
            return None
        return None
    try:
        parsed = int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _require_nonempty(path: Path, operation: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RenderError(
            f"Render command succeeded but {operation} output is missing or empty: "
            f"{path}"
        )


def _srt_time(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

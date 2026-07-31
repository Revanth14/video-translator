from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from dub_mvp.synthesize import SynthesizedSegment


class RenderError(RuntimeError):
    pass


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ToolResolver = Callable[[str], str | None]


class AlignmentSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    duration_budget_ms: int
    tts_audio_path: str
    tts_duration_ms: int
    tempo_ratio: float
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
        return self


class RenderPlan(BaseModel):
    schema_version: int = 1
    duration_ms: int
    segments: list[AlignmentSegment]

    @model_validator(mode="after")
    def validate_segments(self) -> "RenderPlan":
        if self.duration_ms <= 0:
            raise ValueError("Render duration must be positive.")
        if not self.segments:
            raise ValueError("Render plan must contain at least one segment.")
        previous_end = 0
        for segment in self.segments:
            if segment.end_ms > self.duration_ms:
                raise ValueError("Alignment segment exceeds render duration.")
            if segment.start_ms < previous_end:
                raise ValueError("Alignment segments cannot overlap.")
            previous_end = segment.end_ms
        return self


class RenderPipeline:
    def __init__(
        self,
        runner: CommandRunner = subprocess.run,
        resolver: ToolResolver = shutil.which,
    ) -> None:
        self._runner = runner
        self._resolver = resolver

    def run(
        self,
        *,
        synthesized_segments_path: Path,
        source_segment_path: Path,
        run_directory: Path,
        duration_ms: int,
    ) -> tuple[RenderPlan, dict[str, str]]:
        source_segment_path = source_segment_path.expanduser().resolve()
        if not source_segment_path.is_file():
            raise RenderError(f"Source segment is missing: {source_segment_path}")
        ffmpeg = self._required_tool("ffmpeg")
        synthesized_segments = load_synthesized_segments(
            synthesized_segments_path
        )
        plan = build_render_plan(synthesized_segments, duration_ms=duration_ms)

        metadata_directory = run_directory / "metadata"
        subtitles_directory = run_directory / "subtitles"
        working_directory = run_directory / "working"
        output_directory = run_directory / "outputs"
        for directory in (
            metadata_directory,
            subtitles_directory,
            working_directory,
            output_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        plan_path = metadata_directory / "alignment_plan.json"
        srt_path = subtitles_directory / "hi.srt"
        dubbed_audio_path = working_directory / "dubbed_audio.wav"
        video_path = output_directory / "dubbed_video.mp4"

        _write_json(plan_path, plan.model_dump(mode="json"))
        srt_path.write_text(
            build_srt(synthesized_segments),
            encoding="utf-8",
        )

        self._execute(
            _dubbed_audio_command(ffmpeg, plan, dubbed_audio_path),
            "dubbed audio assembly",
        )
        self._execute(
            [
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
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(video_path),
            ],
            "dubbed video mux",
        )

        for output in (dubbed_audio_path, video_path, srt_path, plan_path):
            if not output.is_file():
                raise RenderError(
                    f"Render command succeeded but output is missing: {output}"
                )

        return plan, {
            "alignment_plan": str(plan_path),
            "hindi_srt": str(srt_path),
            "dubbed_audio": str(dubbed_audio_path),
            "dubbed_video": str(video_path),
        }

    def _required_tool(self, name: str) -> str:
        resolved = self._resolver(name)
        if not resolved:
            raise RenderError(f"Required render tool '{name}' was not found.")
        return resolved

    def _execute(
        self, command: Sequence[str], operation: str
    ) -> subprocess.CompletedProcess[str]:
        result = self._runner(
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "no error output"
            raise RenderError(f"Failed {operation}: {detail}")
        return result


def load_synthesized_segments(path: Path) -> list[SynthesizedSegment]:
    if not path.is_file():
        raise RenderError(f"Synthesized segments are missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [SynthesizedSegment.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError) as error:
        raise RenderError(
            f"Unable to read synthesized segments: {path}"
        ) from error
    if not segments:
        raise RenderError(f"Synthesized segments are empty: {path}")
    return segments


def build_render_plan(
    segments: Sequence[SynthesizedSegment],
    *,
    duration_ms: int,
    review_threshold: float = 0.15,
    hard_limit: float = 0.20,
) -> RenderPlan:
    aligned: list[AlignmentSegment] = []
    previous_end = 0
    for segment in segments:
        if segment.start_ms < previous_end:
            raise RenderError(f"Segment overlap at {segment.segment_id}.")
        audio_path = Path(segment.tts_audio_path)
        if not audio_path.is_file():
            raise RenderError(
                f"Synthesized audio is missing for {segment.segment_id}: "
                f"{audio_path}"
            )
        drift = (
            segment.tts_duration_ms - segment.duration_budget_ms
        ) / segment.duration_budget_ms
        if abs(drift) > hard_limit:
            raise RenderError(
                f"{segment.segment_id} requires {abs(drift):.1%} tempo "
                "correction, above the 20% hard limit."
            )
        notes = []
        needs_review = abs(drift) > review_threshold
        if needs_review:
            notes.append(
                f"Tempo correction {abs(drift):.1%} exceeds review threshold."
            )
        aligned.append(
            AlignmentSegment(
                segment_id=segment.segment_id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                duration_budget_ms=segment.duration_budget_ms,
                tts_audio_path=segment.tts_audio_path,
                tts_duration_ms=segment.tts_duration_ms,
                tempo_ratio=round(
                    segment.tts_duration_ms / segment.duration_budget_ms,
                    6,
                ),
                needs_review=needs_review,
                notes=notes,
            )
        )
        previous_end = segment.end_ms
    return RenderPlan(duration_ms=duration_ms, segments=aligned)


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


def _dubbed_audio_command(
    ffmpeg: str,
    plan: RenderPlan,
    output_path: Path,
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for segment in plan.segments:
        command.extend(["-i", segment.tts_audio_path])

    filters = []
    labels = []
    for index, segment in enumerate(plan.segments):
        label = f"a{index}"
        filters.append(
            (
                f"[{index}:a]atempo={segment.tempo_ratio:.6f},"
                f"adelay={segment.start_ms}|{segment.start_ms}[{label}]"
            )
        )
        labels.append(f"[{label}]")
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:"
        f"duration=longest:normalize=0[dub]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[dub]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return command


def _srt_time(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

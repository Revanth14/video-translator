from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dub_mvp.manifest import MediaMetadata
from dub_mvp.timecode import format_timecode_seconds


class MediaToolError(RuntimeError):
    pass


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ToolResolver = Callable[[str], str | None]


class MediaIngestor:
    def __init__(
        self,
        runner: CommandRunner = subprocess.run,
        resolver: ToolResolver = shutil.which,
    ) -> None:
        self._runner = runner
        self._resolver = resolver

    def inspect(self, source: Path) -> MediaMetadata:
        """Inspect source media without creating run artifacts."""
        source = source.expanduser().resolve()
        if not source.is_file():
            raise MediaToolError(f"Input video does not exist: {source}")
        ffprobe = self._required_tool("ffprobe")
        return _metadata_from_probe(self._probe(ffprobe, source))

    def ingest(
        self,
        source: Path,
        run_directory: Path,
        start_ms: int,
        end_ms: int,
    ) -> tuple[MediaMetadata, dict[str, str]]:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise MediaToolError(f"Input video does not exist: {source}")
        if end_ms <= start_ms:
            raise MediaToolError("End time must be greater than start time.")

        ffprobe = self._required_tool("ffprobe")
        ffmpeg = self._required_tool("ffmpeg")
        probe_data = self._probe(ffprobe, source)
        metadata = _metadata_from_probe(probe_data)

        if end_ms > int(metadata.duration_seconds * 1000) + 100:
            raise MediaToolError(
                "Requested end time exceeds the source duration "
                f"({metadata.duration_seconds:.3f}s)."
            )

        metadata_directory = run_directory / "metadata"
        working_directory = run_directory / "working"
        metadata_directory.mkdir(parents=True, exist_ok=True)
        working_directory.mkdir(parents=True, exist_ok=True)

        probe_path = metadata_directory / "ffprobe.json"
        probe_path.write_text(
            json.dumps(probe_data, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        segment_path = working_directory / "source_segment.mp4"
        audio_path = working_directory / "source_audio.wav"
        duration_ms = end_ms - start_ms

        self._execute(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                format_timecode_seconds(start_ms),
                "-i",
                str(source),
                "-t",
                format_timecode_seconds(duration_ms),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                str(segment_path),
            ],
            "source-range extraction",
        )
        self._execute(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(segment_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            "working-audio extraction",
        )

        for output in (segment_path, audio_path):
            if not output.is_file():
                raise MediaToolError(
                    f"Media command succeeded but output is missing: {output}"
                )

        outputs = {
            "probe": str(probe_path),
            "source_segment": str(segment_path),
            "working_audio": str(audio_path),
        }
        return metadata, outputs

    def _required_tool(self, name: str) -> str:
        resolved = self._resolver(name)
        if not resolved:
            raise MediaToolError(
                f"Required media tool '{name}' was not found on PATH."
            )
        return resolved

    def _probe(self, ffprobe: str, source: Path) -> dict[str, Any]:
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ]
        result = self._execute(command, "media inspection")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise MediaToolError("FFprobe returned invalid JSON.") from error

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
            raise MediaToolError(f"Failed {operation}: {detail}")
        return result


def media_duration_ms(metadata: MediaMetadata) -> int:
    duration_ms = int(round(metadata.duration_seconds * 1000))
    if duration_ms <= 0:
        raise MediaToolError("Source duration must be positive.")
    return duration_ms


def _metadata_from_probe(payload: dict[str, Any]) -> MediaMetadata:
    streams = payload.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if not video:
        raise MediaToolError("Input does not contain a video stream.")
    if not audio:
        raise MediaToolError("Input does not contain an audio stream.")

    raw_duration = payload.get("format", {}).get("duration")
    if raw_duration is None:
        raw_duration = video.get("duration") or audio.get("duration")
    try:
        duration_seconds = float(raw_duration)
    except (TypeError, ValueError) as error:
        raise MediaToolError("Unable to determine source duration.") from error

    return MediaMetadata(
        duration_seconds=duration_seconds,
        format_name=payload.get("format", {}).get("format_name"),
        video_codec=str(video.get("codec_name", "unknown")),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        frame_rate=video.get("avg_frame_rate"),
        audio_codec=str(audio.get("codec_name", "unknown")),
        audio_channels=_optional_int(audio.get("channels")),
        audio_sample_rate=_optional_int(audio.get("sample_rate")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

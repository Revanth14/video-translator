import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from dub_mvp.media import MediaIngestor, MediaToolError


PROBE_PAYLOAD = {
    "format": {
        "duration": "120.5",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
    },
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30/1",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 2,
            "sample_rate": "48000",
        },
    ],
}


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[0] == "/fake/ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(PROBE_PAYLOAD),
                stderr="",
            )

        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def resolver(name: str) -> str:
    return f"/fake/{name}"


def test_ingest_creates_expected_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    run_directory = tmp_path / "run"
    runner = FakeRunner()

    metadata, outputs = MediaIngestor(
        runner=runner,
        resolver=resolver,
    ).ingest(source, run_directory, 1000, 11000)

    assert metadata.duration_seconds == 120.5
    assert metadata.video_codec == "h264"
    assert metadata.audio_sample_rate == 48000
    assert Path(outputs["probe"]).is_file()
    assert Path(outputs["source_segment"]).is_file()
    assert Path(outputs["working_audio"]).is_file()
    assert len(runner.commands) == 3
    assert "-ss" in runner.commands[1]
    assert "1.000" in runner.commands[1]
    assert "10.000" in runner.commands[1]


def test_ingest_reports_missing_media_tools(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()

    with pytest.raises(MediaToolError, match="ffprobe"):
        MediaIngestor(resolver=lambda _: None).ingest(
            source,
            tmp_path / "run",
            0,
            1000,
        )


def test_ingest_rejects_end_after_source_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()

    with pytest.raises(MediaToolError, match="exceeds"):
        MediaIngestor(
            runner=FakeRunner(),
            resolver=resolver,
        ).ingest(source, tmp_path / "run", 0, 121000)

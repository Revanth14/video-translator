from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    INGESTED = "ingested"
    TRANSCRIBED = "transcribed"
    LOCALIZED = "localized"
    SYNTHESIZED = "synthesized"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class MediaMetadata(BaseModel):
    duration_seconds: float
    format_name: str | None = None
    video_codec: str
    width: int
    height: int
    frame_rate: str | None = None
    audio_codec: str
    audio_channels: int | None = None
    audio_sample_rate: int | None = None


class RunManifest(BaseModel):
    schema_version: int = 1
    run_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_path: str
    source_start_ms: int
    source_end_ms: int
    source_language: str = "en"
    target_language: str = "hi"
    status: RunStatus = RunStatus.CREATED
    media: MediaMetadata | None = None
    models: dict[str, str] = Field(default_factory=dict)
    stages: dict[str, StageRecord] = Field(
        default_factory=lambda: {
            "ingest": StageRecord(),
            "transcribe": StageRecord(),
            "localize": StageRecord(),
            "synthesize": StageRecord(),
        }
    )
    outputs: dict[str, str] = Field(default_factory=dict)
    timings_seconds: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.source_end_ms - self.source_start_ms

    def save(self, run_directory: Path) -> Path:
        run_directory.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc)
        manifest_path = run_directory / "manifest.json"
        temporary_path = run_directory / ".manifest.json.tmp"
        payload = self.model_dump(mode="json")

        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, manifest_path)
        return manifest_path

    @classmethod
    def load(cls, run_directory: Path) -> "RunManifest":
        manifest_path = run_directory / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))

    def public_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "source": self.source_path,
            "range_ms": [self.source_start_ms, self.source_end_ms],
            "target_language": self.target_language,
            "stages": {
                name: stage.status.value for name, stage in self.stages.items()
            },
            "outputs": self.outputs,
            "errors": self.errors,
        }

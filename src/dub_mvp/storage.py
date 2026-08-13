from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field


DEFAULT_MINIMUM_FREE_BYTES = 512 * 1024 * 1024

# These are conservative scratch-space guards, not capacity promises. A real
# long-form benchmark must replace them with measured per-stage high-water
# marks before remote instance storage is selected.
_SOURCE_SIZE_MULTIPLIERS = {
    "ingest": 1.25,
    "transcribe": 0.5,
    "segment": 0.1,
    "localize": 0.1,
    "synthesize": 0.75,
    "render": 2.0,
}


class StorageCapacityError(RuntimeError):
    retryable = True


class StorageCapacity(BaseModel):
    stage: str
    free_bytes: int = Field(ge=0)
    required_bytes: int = Field(ge=0)
    source_size_bytes: int = Field(ge=0)

    @property
    def sufficient(self) -> bool:
        return self.free_bytes >= self.required_bytes


def measure_stage_capacity(
    run_directory: Path,
    *,
    stage: str,
    source_path: Path,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    free_bytes: Callable[[Path], int] | None = None,
) -> StorageCapacity:
    if minimum_free_bytes < 0:
        raise ValueError("Minimum free bytes cannot be negative.")
    if stage not in _SOURCE_SIZE_MULTIPLIERS:
        raise ValueError(f"Unknown storage stage: {stage}")
    try:
        source_size = source_path.stat().st_size if source_path.is_file() else 0
    except OSError:
        source_size = 0
    required = max(
        minimum_free_bytes,
        int(source_size * _SOURCE_SIZE_MULTIPLIERS[stage]),
    )
    available = (
        free_bytes(run_directory)
        if free_bytes is not None
        else shutil.disk_usage(run_directory).free
    )
    return StorageCapacity(
        stage=stage,
        free_bytes=max(0, int(available)),
        required_bytes=required,
        source_size_bytes=source_size,
    )


def require_stage_capacity(
    run_directory: Path,
    *,
    stage: str,
    source_path: Path,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    free_bytes: Callable[[Path], int] | None = None,
) -> StorageCapacity:
    capacity = measure_stage_capacity(
        run_directory,
        stage=stage,
        source_path=source_path,
        minimum_free_bytes=minimum_free_bytes,
        free_bytes=free_bytes,
    )
    if not capacity.sufficient:
        raise StorageCapacityError(
            f"Insufficient disk space for {stage}: "
            f"{capacity.free_bytes} bytes free, "
            f"{capacity.required_bytes} bytes required."
        )
    return capacity

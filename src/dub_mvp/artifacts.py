from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ArtifactStatus(str, Enum):
    COMPLETED = "completed"
    INVALID = "invalid"


class ArtifactMetadata(BaseModel):
    """Sidecar proving an artifact is complete and matches its inputs.

    `path` is always relative to the run directory so a run stays verifiable
    after it is moved, copied, or uploaded to object storage.
    """

    schema_version: int = Field(default=1, ge=1)
    artifact_id: str
    kind: str
    status: ArtifactStatus
    path: str
    output_sha256: str
    input_fingerprint: str
    size_bytes: int = Field(ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provider: str | None = None
    model: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "artifact_id",
        "kind",
        "path",
        "output_sha256",
        "input_fingerprint",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Artifact metadata fields cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_hashes(self) -> "ArtifactMetadata":
        for name, value in (
            ("output_sha256", self.output_sha256),
            ("input_fingerprint", self.input_fingerprint),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        return self


class ArtifactVerification(BaseModel):
    valid: bool
    reason: str | None = None


def fingerprint_inputs(payload: Any) -> str:
    """Hash the inputs that decide whether an artifact can be reused.

    Only include values that should force regeneration when they change. A
    timestamp would differ on every call and defeat reuse entirely, so bare
    datetimes are rejected. Note this cannot catch a timestamp nested inside a
    model, which serializes to a string before reaching the encoder.
    """
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_artifact_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(
            f"Artifact {path} is outside its run directory {root}."
        ) from error


def completed_artifact_metadata(
    *,
    artifact_id: str,
    kind: str,
    path: Path,
    root: Path,
    inputs: Any,
    provider: str | None = None,
    model: str | None = None,
    configuration: dict[str, Any] | None = None,
) -> ArtifactMetadata:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    return ArtifactMetadata(
        artifact_id=artifact_id,
        kind=kind,
        status=ArtifactStatus.COMPLETED,
        path=relative_artifact_path(path, root),
        output_sha256=sha256_file(path),
        input_fingerprint=fingerprint_inputs(inputs),
        size_bytes=path.stat().st_size,
        provider=provider,
        model=model,
        configuration=configuration or {},
    )


def verify_artifact(
    metadata: ArtifactMetadata,
    *,
    expected_inputs: Any,
    root: Path,
) -> ArtifactVerification:
    if metadata.status != ArtifactStatus.COMPLETED:
        return ArtifactVerification(valid=False, reason="artifact is not completed")
    if metadata.input_fingerprint != fingerprint_inputs(expected_inputs):
        return ArtifactVerification(valid=False, reason="input fingerprint mismatch")

    path = root / metadata.path
    if not path.is_file():
        return ArtifactVerification(valid=False, reason="artifact file is missing")
    if path.stat().st_size != metadata.size_bytes:
        return ArtifactVerification(valid=False, reason="artifact size mismatch")
    if sha256_file(path) != metadata.output_sha256:
        return ArtifactVerification(valid=False, reason="artifact checksum mismatch")
    return ArtifactVerification(valid=True)


def write_artifact_metadata(path: Path, metadata: ArtifactMetadata) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = metadata.model_dump(mode="json")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        raise TypeError(
            "Timestamps cannot take part in an input fingerprint: they change "
            "on every call, so no artifact would ever be reusable."
        )
    raise TypeError(f"Unsupported fingerprint input: {type(value).__name__}")


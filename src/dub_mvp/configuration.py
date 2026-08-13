from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from dub_mvp.artifacts import (
    completed_artifact_metadata,
    relative_artifact_path,
    sha256_file,
    write_artifact_metadata,
)
from dub_mvp.duration import DurationCorrector
from dub_mvp.render import RenderPolicy


SUPPORTED_RELEASE_LANGUAGE_PAIRS = {("en", "hi")}


class ConfigurationError(ValueError):
    pass


class PipelineConfigurationSnapshot(BaseModel):
    schema_version: int = 1
    source_language: str
    target_language: str
    translation_provider: str
    translation_model: str
    tts_provider: str
    tts_model: str
    voice_catalog_path: str | None = None
    voice_catalog_sha256: str | None = None
    glossary_path: str | None = None
    glossary_sha256: str | None = None
    translation_context_path: str | None = None
    translation_context_sha256: str | None = None
    duration_policy_fingerprint: str
    render_policy_fingerprint: str

    @field_validator(
        "source_language",
        "target_language",
        "translation_provider",
        "translation_model",
        "tts_provider",
        "tts_model",
        "duration_policy_fingerprint",
        "render_policy_fingerprint",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Pipeline configuration fields cannot be empty.")
        return cleaned


def validate_release_language_pair(
    source_language: str,
    target_language: str,
) -> tuple[str, str]:
    source = source_language.strip().lower()
    target = target_language.strip().lower()
    language_pattern = r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*"
    if not re.fullmatch(language_pattern, source) or not re.fullmatch(
        language_pattern, target
    ):
        raise ConfigurationError("Language codes must be BCP-47-style values.")
    if (source, target) not in SUPPORTED_RELEASE_LANGUAGE_PAIRS:
        raise ConfigurationError(
            f"Language pair {source}->{target} is not release-enabled. "
            "Run language-check with a passing Hindi baseline and a candidate "
            "evaluation set before expanding the configured registry."
        )
    return source, target


def build_configuration_snapshot(
    *,
    run_directory: Path,
    source_language: str,
    target_language: str,
    translation_pipeline: Any | None = None,
    synthesis_pipeline: Any | None = None,
    render_pipeline: Any | None = None,
    voice_catalog_path: Path | None = None,
    glossary_path: Path | None = None,
    translation_context_path: Path | None = None,
) -> PipelineConfigurationSnapshot:
    source, target = validate_release_language_pair(
        source_language, target_language
    )
    translation_provider, translation_model = _provider_identity(
        translation_pipeline,
        default_provider="openai",
        default_model="gpt-5-mini",
    )
    synthesis_provider, synthesis_model = _provider_identity(
        synthesis_pipeline,
        default_provider="indicf5",
        default_model="ai4bharat/IndicF5",
    )
    duration_corrector = getattr(synthesis_pipeline, "duration_corrector", None)
    duration_fingerprint = getattr(
        duration_corrector,
        "configuration_fingerprint",
        DurationCorrector().configuration_fingerprint,
    )
    render_policy = getattr(render_pipeline, "policy", None) or RenderPolicy()
    render_fingerprint = getattr(
        render_policy,
        "configuration_fingerprint",
        RenderPolicy().configuration_fingerprint,
    )
    return PipelineConfigurationSnapshot(
        source_language=source,
        target_language=target,
        translation_provider=translation_provider,
        translation_model=translation_model,
        tts_provider=synthesis_provider,
        tts_model=synthesis_model,
        voice_catalog_path=_relative_existing(voice_catalog_path, run_directory),
        voice_catalog_sha256=_checksum(voice_catalog_path),
        glossary_path=_relative_existing(glossary_path, run_directory),
        glossary_sha256=_checksum(glossary_path),
        translation_context_path=_relative_existing(
            translation_context_path, run_directory
        ),
        translation_context_sha256=_checksum(translation_context_path),
        duration_policy_fingerprint=duration_fingerprint,
        render_policy_fingerprint=render_fingerprint,
    )


def write_configuration_snapshot(
    snapshot: PipelineConfigurationSnapshot,
    *,
    run_directory: Path,
) -> dict[str, str]:
    path = run_directory / "input" / "pipeline-config.json"
    metadata_path = path.with_name(path.name + ".meta.json")
    _write_json(path, snapshot.model_dump(mode="json"))
    write_artifact_metadata(
        metadata_path,
        completed_artifact_metadata(
            artifact_id="pipeline_configuration",
            kind="pipeline_configuration",
            path=path,
            root=run_directory,
            inputs=snapshot.model_dump(mode="json"),
            provider="internal",
        ),
    )
    return {
        "configuration_snapshot": str(path.resolve()),
        "configuration_snapshot_metadata": str(metadata_path.resolve()),
    }


def _provider_identity(
    pipeline: Any | None,
    *,
    default_provider: str,
    default_model: str,
) -> tuple[str, str]:
    if pipeline is None:
        return default_provider, default_model
    provider = getattr(pipeline, "_provider", pipeline)
    name = getattr(provider, "provider_name", type(provider).__name__)
    model = getattr(provider, "model_name", "injected")
    return str(name), str(model)


def _relative_existing(path: Path | None, root: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    return relative_artifact_path(path, root)


def _checksum(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)

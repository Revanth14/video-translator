from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dub_mvp.manifest import RunManifest
from dub_mvp.synthesize import SynthesisError, load_voice_reference


class PreflightCheck(BaseModel):
    name: str
    status: str
    detail: str


class PreflightReport(BaseModel):
    ok: bool
    checks: list[PreflightCheck]


def build_preflight_report(
    *,
    run_directory: Path | None = None,
    voice_reference_path: Path | None = None,
) -> PreflightReport:
    checks = [
        _tool_check("ffmpeg"),
        _tool_check("ffprobe"),
        _module_check("whisperx", required=False),
        _module_check("openai", required=False),
        _module_check("indicf5", required=False),
        _env_check("OPENAI_API_KEY", required=False),
    ]

    if run_directory is not None:
        checks.extend(_run_checks(run_directory))
    if voice_reference_path is not None:
        checks.append(_voice_reference_check(voice_reference_path))

    blocking_statuses = {"fail"}
    return PreflightReport(
        ok=not any(check.status in blocking_statuses for check in checks),
        checks=checks,
    )


def report_to_json(report: PreflightReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2) + "\n"


def _tool_check(name: str) -> PreflightCheck:
    resolved = shutil.which(name)
    if resolved:
        return PreflightCheck(
            name=f"tool:{name}",
            status="pass",
            detail=resolved,
        )
    return PreflightCheck(
        name=f"tool:{name}",
        status="fail",
        detail=f"{name} was not found on PATH.",
    )


def _module_check(name: str, *, required: bool) -> PreflightCheck:
    if importlib.util.find_spec(name) is not None:
        return PreflightCheck(
            name=f"python:{name}",
            status="pass",
            detail=f"{name} is importable.",
        )
    return PreflightCheck(
        name=f"python:{name}",
        status="fail" if required else "warn",
        detail=f"{name} is not installed in this Python environment.",
    )


def _env_check(name: str, *, required: bool) -> PreflightCheck:
    if os.environ.get(name):
        return PreflightCheck(
            name=f"env:{name}",
            status="pass",
            detail=f"{name} is set.",
        )
    return PreflightCheck(
        name=f"env:{name}",
        status="fail" if required else "warn",
        detail=f"{name} is not set.",
    )


def _run_checks(run_directory: Path) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    try:
        manifest = RunManifest.load(run_directory)
    except (OSError, ValueError) as error:
        return [
            PreflightCheck(
                name="run:manifest",
                status="fail",
                detail=f"Unable to read manifest: {error}",
            )
        ]

    checks.append(
        PreflightCheck(
            name="run:manifest",
            status="pass",
            detail=f"Loaded run {manifest.run_id}.",
        )
    )
    expected_outputs = [
        "source_segment",
        "working_audio",
        "segments",
        "dubbing_utterances",
        "translation_segments",
        "localized_segments",
        "synthesized_segments",
    ]
    for output_name in expected_outputs:
        path = manifest.outputs.get(output_name)
        if path and Path(path).is_file():
            checks.append(
                PreflightCheck(
                    name=f"run:output:{output_name}",
                    status="pass",
                    detail=path,
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name=f"run:output:{output_name}",
                    status="warn",
                    detail=f"{output_name} has not been produced yet.",
                )
            )
    return checks


def _voice_reference_check(path: Path) -> PreflightCheck:
    try:
        reference = load_voice_reference(path)
    except SynthesisError as error:
        return PreflightCheck(
            name="voice_reference",
            status="fail",
            detail=str(error),
        )
    return PreflightCheck(
        name="voice_reference",
        status="pass",
        detail=(
            f"Loaded {reference.reference_id}; consent: {reference.consent}"
        ),
    )

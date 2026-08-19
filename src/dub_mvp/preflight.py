from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dub_mvp.manifest import RunManifest
from dub_mvp.media import MediaIngestor, MediaToolError
from dub_mvp.synthesize import SynthesisError, load_voice_catalog
from dub_mvp.storage import measure_stage_capacity


MINIMUM_BENCHMARK_DURATION_SECONDS = 30 * 60
MAXIMUM_BENCHMARK_DURATION_SECONDS = 45 * 60


class PreflightProfile(str, Enum):
    LOCAL = "local"
    BENCHMARK = "benchmark"


class PreflightCheck(BaseModel):
    name: str
    status: str
    detail: str


class PreflightReport(BaseModel):
    profile: PreflightProfile = PreflightProfile.LOCAL
    ok: bool
    checks: list[PreflightCheck]


def build_preflight_report(
    *,
    profile: PreflightProfile = PreflightProfile.LOCAL,
    run_directory: Path | None = None,
    voice_reference_path: Path | None = None,
    input_video_path: Path | None = None,
    media_ingestor: MediaIngestor | None = None,
) -> PreflightReport:
    benchmark = profile == PreflightProfile.BENCHMARK
    checks = [
        _tool_check("ffmpeg"),
        _tool_check("ffprobe"),
        _module_check("whisperx", required=benchmark),
        _module_check("openai", required=benchmark),
        _module_check("indicf5", required=benchmark),
        _module_check("torch", required=benchmark),
        _env_check("OPENAI_API_KEY", required=benchmark),
    ]

    if benchmark:
        checks.extend(
            [
                _tool_check("nvidia-smi"),
                _cuda_check(),
                _translation_pricing_check(),
                _benchmark_input_check(
                    input_video_path,
                    ingestor=media_ingestor or MediaIngestor(),
                ),
            ]
        )

    if run_directory is not None:
        checks.extend(_run_checks(run_directory))
    if voice_reference_path is not None:
        checks.append(_voice_reference_check(voice_reference_path))
    elif benchmark:
        checks.append(
            PreflightCheck(
                name="voice_reference",
                status="fail",
                detail=(
                    "Benchmark preflight requires --voice-reference with a "
                    "validated consented voice catalog."
                ),
            )
        )

    blocking_statuses = {"fail"}
    return PreflightReport(
        profile=profile,
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


def _cuda_check() -> PreflightCheck:
    try:
        torch = importlib.import_module("torch")
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("torch.cuda.is_available() returned false")
        device_count = int(torch.cuda.device_count())
        devices = [
            str(torch.cuda.get_device_name(index))
            for index in range(device_count)
        ]
    except Exception as error:
        return PreflightCheck(
            name="runtime:cuda",
            status="fail",
            detail=f"CUDA is unavailable to Torch: {error}",
        )
    return PreflightCheck(
        name="runtime:cuda",
        status="pass",
        detail=(
            f"Torch sees {device_count} CUDA device(s): "
            + ", ".join(devices)
        ),
    )


def _translation_pricing_check() -> PreflightCheck:
    names = (
        "VIDEO_TRANSLATOR_OPENAI_INPUT_USD_PER_MILLION",
        "VIDEO_TRANSLATOR_OPENAI_OUTPUT_USD_PER_MILLION",
    )
    values: list[float] = []
    try:
        for name in names:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                raise ValueError(f"{name} is not set")
            value = float(raw)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            values.append(value)
    except ValueError as error:
        return PreflightCheck(
            name="env:translation_pricing",
            status="fail",
            detail=(
                "Benchmark cost reporting requires current non-negative "
                f"translation token prices: {error}."
            ),
        )
    return PreflightCheck(
        name="env:translation_pricing",
        status="pass",
        detail=(
            "Configured translation prices per million tokens: "
            f"input={values[0]}, output={values[1]}."
        ),
    )


def _benchmark_input_check(
    path: Path | None,
    *,
    ingestor: MediaIngestor,
) -> PreflightCheck:
    if path is None:
        return PreflightCheck(
            name="benchmark:input",
            status="fail",
            detail=(
                "Benchmark preflight requires --input-video with an "
                "authorized 30-45 minute source."
            ),
        )
    try:
        metadata = ingestor.inspect(path)
    except (MediaToolError, OSError, ValueError) as error:
        return PreflightCheck(
            name="benchmark:input",
            status="fail",
            detail=f"Unable to inspect benchmark input: {error}",
        )
    if not (
        MINIMUM_BENCHMARK_DURATION_SECONDS
        <= metadata.duration_seconds
        <= MAXIMUM_BENCHMARK_DURATION_SECONDS
    ):
        return PreflightCheck(
            name="benchmark:input",
            status="fail",
            detail=(
                "Benchmark input duration must be 30-45 minutes; found "
                f"{metadata.duration_seconds / 60:.2f} minutes."
            ),
        )
    return PreflightCheck(
        name="benchmark:input",
        status="pass",
        detail=(
            f"{metadata.duration_seconds / 60:.2f} minutes, "
            f"{metadata.width}x{metadata.height}, "
            f"video={metadata.video_codec}, audio={metadata.audio_codec}."
        ),
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
    for stage_name, record in manifest.stages.items():
        if record.status.value not in {"queued", "running"}:
            continue
        try:
            capacity = measure_stage_capacity(
                run_directory,
                stage=stage_name,
                source_path=Path(manifest.source_path),
            )
        except (OSError, ValueError) as error:
            checks.append(
                PreflightCheck(
                    name=f"run:disk:{stage_name}",
                    status="fail",
                    detail=f"Unable to measure disk capacity: {error}",
                )
            )
            continue
        checks.append(
            PreflightCheck(
                name=f"run:disk:{stage_name}",
                status="pass" if capacity.sufficient else "fail",
                detail=(
                    f"{capacity.free_bytes} bytes free; "
                    f"{capacity.required_bytes} bytes required."
                ),
            )
        )
    expected_outputs = [
        "source_segment",
        "working_audio",
        "segments",
        "dubbing_utterances",
        "translation_segments",
        "localized_segments",
        "localized_segments_metadata",
        "translation_metrics",
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
        catalog = load_voice_catalog(path)
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
            f"Loaded {len(catalog.voices)} consented voice(s): "
            + ", ".join(voice.reference_id for voice in catalog.voices)
        ),
    )

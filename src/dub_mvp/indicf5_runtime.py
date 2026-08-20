from __future__ import annotations

import gc
import json
import os
import sys
import traceback
import wave
from pathlib import Path
from typing import Any


class RuntimeConfigurationError(RuntimeError):
    """The isolated runtime cannot run with its installed configuration."""


def _read_request(path: Path) -> dict[str, Any]:
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeConfigurationError("Unable to read IndicF5 request.") from error
    required = {
        "schema_version",
        "translated_text",
        "tts_text",
        "text_normalization_policy",
        "text_batches",
        "output_path",
        "reference_audio",
        "reference_text",
        "reference_seconds",
        "fix_duration_seconds",
    }
    if not isinstance(request, dict) or not required.issubset(request):
        raise RuntimeConfigurationError("IndicF5 request is missing required fields.")
    if request["schema_version"] != 4:
        raise RuntimeConfigurationError(
            "Unsupported IndicF5 request schema; expected version 4."
        )
    for key in ("translated_text", "tts_text", "text_normalization_policy"):
        if not isinstance(request[key], str) or not request[key].strip():
            raise RuntimeConfigurationError(
                f"IndicF5 request contains invalid {key}."
            )
    return request


def _validate_batches(request: dict[str, Any]) -> list[str]:
    batches = request["text_batches"]
    if not isinstance(batches, list) or not batches:
        raise RuntimeConfigurationError("IndicF5 text batches cannot be empty.")
    if not all(isinstance(batch, str) and batch.strip() for batch in batches):
        raise RuntimeConfigurationError("IndicF5 text batches contain empty text.")
    normalized_batches = [" ".join(batch.split()) for batch in batches]
    normalized_target = " ".join(str(request["tts_text"]).split())
    if " ".join(normalized_batches) != normalized_target:
        raise RuntimeConfigurationError("IndicF5 text batches alter the TTS text.")
    if len(normalized_batches) != 1:
        # fix_duration is applied per batch, so it only describes the whole
        # utterance while the utterance is a single batch.
        raise RuntimeConfigurationError(
            "Duration-pinned IndicF5 synthesis requires exactly one text batch; "
            f"got {len(normalized_batches)}."
        )
    return normalized_batches


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _positive_float(request: dict[str, Any], key: str) -> float:
    try:
        value = float(request[key])
    except (TypeError, ValueError) as error:
        raise RuntimeConfigurationError(f"Invalid IndicF5 {key}.") from error
    if not value > 0 or value != value or value in {float("inf"), float("-inf")}:
        raise RuntimeConfigurationError(f"IndicF5 {key} must be a positive number.")
    return value


def _synthesize(request: dict[str, Any]) -> dict[str, Any]:
    try:
        import inspect

        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from f5_tts.infer.utils_infer import (
            infer_batch_process,
            load_model,
            load_vocoder,
            preprocess_ref_audio_text,
        )
        from f5_tts.model import DiT
        from safetensors.torch import load_file
        from transformers import PreTrainedModel, PretrainedConfig
    except ImportError as error:
        raise RuntimeConfigurationError(
            f"IndicF5 dependency is unavailable: {error.name}"
        ) from error

    model_root = Path(
        os.environ.get(
            "VIDEO_TRANSLATOR_INDICF5_MODEL_ROOT",
            "/opt/video-translator/model-cache/indicf5-artifacts",
        )
    )
    vocoder_root = Path(
        os.environ.get(
            "VIDEO_TRANSLATOR_INDICF5_VOCODER_ROOT",
            "/opt/video-translator/model-cache/vocos-mel-24khz",
        )
    )
    model_path = model_root / "model.safetensors"
    vocab_path = model_root / "checkpoints" / "vocab.txt"
    for path in (model_path, vocab_path, vocoder_root):
        if not path.exists():
            raise RuntimeConfigurationError(f"IndicF5 artifact is missing: {path}")

    reference_audio = Path(str(request["reference_audio"]))
    output_path = Path(str(request["output_path"]))
    if not reference_audio.is_file():
        raise RuntimeConfigurationError(
            f"IndicF5 reference audio is missing: {reference_audio}"
        )
    batches = _validate_batches(request)
    expected_reference_seconds = _positive_float(request, "reference_seconds")
    fix_duration_seconds = _positive_float(request, "fix_duration_seconds")
    if fix_duration_seconds <= expected_reference_seconds:
        raise RuntimeConfigurationError(
            "IndicF5 fix_duration_seconds must exceed the reference duration; "
            "it describes the whole reference-plus-generated window."
        )
    if "fix_duration" not in inspect.signature(infer_batch_process).parameters:
        raise RuntimeConfigurationError(
            "The installed f5_tts does not accept fix_duration. Without it "
            "generation length falls back to a UTF-8 byte ratio, which "
            "mistimes non-Latin output."
        )

    class OfflineIndicF5Config(PretrainedConfig):
        model_type = "inf5"

        def __init__(self, speed: float = 1.0, remove_sil: bool = True, **kwargs):
            super().__init__(**kwargs)
            self.speed = speed
            self.remove_sil = remove_sil

    class OfflineIndicF5Model(PreTrainedModel):
        config_class = OfflineIndicF5Config

        def __init__(self, config):
            super().__init__(config)
            device = torch.device("cuda")
            self.vocoder = torch.compile(
                load_vocoder(
                    vocoder_name="vocos",
                    is_local=True,
                    local_path=str(vocoder_root),
                    device=device,
                )
            )
            self.ema_model = torch.compile(
                load_model(
                    DiT,
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        conv_layers=4,
                    ),
                    mel_spec_type="vocos",
                    vocab_file=str(vocab_path),
                    device=device,
                )
            )

    if not torch.cuda.is_available():
        raise RuntimeConfigurationError("IndicF5 CUDA device is unavailable.")
    model = OfflineIndicF5Model(OfflineIndicF5Config())
    state = load_file(str(model_path), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()

    ref_audio, normalized_reference_text = preprocess_ref_audio_text(
        str(reference_audio), str(request["reference_text"])
    )
    reference_waveform, reference_sample_rate = torchaudio.load(ref_audio)
    # Preprocessing trims edge silence and clips over-long clips, so the length
    # the caller budgeted against is not necessarily the one the model is
    # prompted with. Small shifts are absorbed by the arithmetic below. A large
    # shift means the clip was cut, so the transcript no longer describes the
    # audio and the speaking rate the budget was calibrated from no longer holds.
    actual_reference_seconds = reference_waveform.shape[-1] / reference_sample_rate
    if abs(actual_reference_seconds - expected_reference_seconds) > 2.0:
        raise RuntimeConfigurationError(
            f"IndicF5 preprocessing changed the reference from "
            f"{expected_reference_seconds:.2f}s to "
            f"{actual_reference_seconds:.2f}s. The transcript and duration "
            "budget describe the original clip, so neither still applies."
        )
    # fix_duration is the whole conditioning window and the reference span is
    # sliced back off, so restate it against the reference actually loaded to
    # keep the generated portion equal to the caller's timeline budget.
    target_seconds = fix_duration_seconds - expected_reference_seconds
    audio, sample_rate, _ = infer_batch_process(
        (reference_waveform, reference_sample_rate),
        normalized_reference_text,
        batches,
        model.ema_model,
        model.vocoder,
        mel_spec_type="vocos",
        speed=1.0,
        fix_duration=actual_reference_seconds + target_seconds,
        device="cuda",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    sf.write(
        str(temporary),
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        format="WAV",
    )
    try:
        with wave.open(str(temporary), "rb") as handle:
            frames = handle.getnframes()
            frame_rate = handle.getframerate()
            channels = handle.getnchannels()
        if frames <= 0 or frame_rate != 24_000 or channels != 1:
            raise RuntimeError("IndicF5 produced invalid WAV parameters.")
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "schema_version": 2,
        "status": "completed",
        "duration_ms": max(1, int(round(frames / frame_rate * 1000))),
        "seed": None,
        "batch_count": len(batches),
        "reference_seconds": round(actual_reference_seconds, 3),
        "fix_duration_seconds": round(fix_duration_seconds, 3),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: indicf5_runtime.py REQUEST RESPONSE", file=sys.stderr)
        return 2
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        response = _synthesize(_read_request(request_path))
        _write_json_atomic(response_path, response)
    except RuntimeConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

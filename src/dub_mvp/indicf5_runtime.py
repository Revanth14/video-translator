from __future__ import annotations

import gc
import json
import os
import sys
import traceback
import wave
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any


RUNTIME_PROTOCOL_VERSION = "stage_ndjson_v1"
RUNTIME_IMPLEMENTATION_REVISION = "indicf5_runtime_v5"
REQUEST_SCHEMA_VERSION = 5


class RuntimeConfigurationError(RuntimeError):
    """The isolated runtime cannot run with its installed configuration."""


def _read_request(path: Path) -> dict[str, Any]:
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeConfigurationError("Unable to read IndicF5 request.") from error
    return _validate_request(request)


def _validate_request(
    request: Any,
    *,
    require_request_id: bool = False,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "model",
        "model_revision",
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
    if require_request_id:
        required.add("request_id")
    if not isinstance(request, dict) or not required.issubset(request):
        raise RuntimeConfigurationError("IndicF5 request is missing required fields.")
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise RuntimeConfigurationError(
            "Unsupported IndicF5 request schema; expected version "
            f"{REQUEST_SCHEMA_VERSION}."
        )
    for key in (
        "model",
        "model_revision",
        "translated_text",
        "tts_text",
        "text_normalization_policy",
    ):
        if not isinstance(request[key], str) or not request[key].strip():
            raise RuntimeConfigurationError(
                f"IndicF5 request contains invalid {key}."
            )
    if require_request_id and (
        not isinstance(request["request_id"], str)
        or not request["request_id"].strip()
    ):
        raise RuntimeConfigurationError(
            "IndicF5 request contains invalid request_id."
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


@dataclass(frozen=True)
class _LoadedRuntime:
    np: Any
    sf: Any
    torchaudio: Any
    infer_batch_process: Any
    preprocess_ref_audio_text: Any
    model: Any


def _load_runtime() -> _LoadedRuntime:
    """Load and compile the expensive model once for this child process."""

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

    return _LoadedRuntime(
        np=np,
        sf=sf,
        torchaudio=torchaudio,
        infer_batch_process=infer_batch_process,
        preprocess_ref_audio_text=preprocess_ref_audio_text,
        model=model,
    )


def _synthesize(
    request: dict[str, Any],
    *,
    runtime: _LoadedRuntime | None = None,
) -> dict[str, Any]:
    runtime = runtime or _load_runtime()

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

    ref_audio, normalized_reference_text = runtime.preprocess_ref_audio_text(
        str(reference_audio), str(request["reference_text"])
    )
    reference_waveform, reference_sample_rate = runtime.torchaudio.load(ref_audio)
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
    audio, sample_rate, _ = runtime.infer_batch_process(
        (reference_waveform, reference_sample_rate),
        normalized_reference_text,
        batches,
        runtime.model.ema_model,
        runtime.model.vocoder,
        mel_spec_type="vocos",
        speed=1.0,
        fix_duration=actual_reference_seconds + target_seconds,
        device="cuda",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    runtime.sf.write(
        str(temporary),
        runtime.np.asarray(audio, dtype=runtime.np.float32),
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


def _emit_protocol(payload: dict[str, Any], output_stream: Any) -> None:
    output_stream.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    output_stream.write("\n")
    output_stream.flush()


def _error_detail(error: BaseException) -> str:
    detail = " ".join(str(error).split()) or type(error).__name__
    return shorten(detail, width=1000, placeholder="...")


def _serve(
    *,
    input_stream: Any = None,
    output_stream: Any = None,
    runtime_loader: Any = None,
    synthesizer: Any = None,
) -> int:
    """Serve sequential correlated requests while keeping one model loaded."""

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    runtime_loader = runtime_loader or _load_runtime
    synthesizer = synthesizer or _synthesize
    try:
        # Provider libraries are not part of the protocol. Anything they print
        # belongs on stderr so stdout remains strict newline-delimited JSON.
        with redirect_stdout(sys.stderr):
            runtime = runtime_loader()
    except RuntimeConfigurationError as error:
        _emit_protocol(
            {
                "type": "ready",
                "status": "failed",
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "runtime_revision": RUNTIME_IMPLEMENTATION_REVISION,
                "retryable": False,
                "error_class": type(error).__name__,
                "error": _error_detail(error),
            },
            output_stream,
        )
        return 2
    except BaseException as error:
        traceback.print_exc(file=sys.stderr)
        _emit_protocol(
            {
                "type": "ready",
                "status": "failed",
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "runtime_revision": RUNTIME_IMPLEMENTATION_REVISION,
                "retryable": True,
                "error_class": type(error).__name__,
                "error": _error_detail(error),
            },
            output_stream,
        )
        return 1

    _emit_protocol(
        {
            "type": "ready",
            "status": "ready",
            "protocol_version": RUNTIME_PROTOCOL_VERSION,
            "runtime_revision": RUNTIME_IMPLEMENTATION_REVISION,
            "request_schema_version": REQUEST_SCHEMA_VERSION,
        },
        output_stream,
    )
    seen_request_ids: set[str] = set()
    for line in input_stream:
        if not line.strip():
            continue
        request_id: str | None = None
        try:
            try:
                request = json.loads(line)
            except (ValueError, TypeError) as error:
                raise RuntimeConfigurationError(
                    "IndicF5 protocol request is not valid JSON."
                ) from error
            validated = _validate_request(request, require_request_id=True)
            request_id = validated["request_id"]
            if request_id in seen_request_ids:
                raise RuntimeConfigurationError(
                    f"IndicF5 request_id was reused: {request_id}."
                )
            seen_request_ids.add(request_id)
            with redirect_stdout(sys.stderr):
                response = synthesizer(validated, runtime=runtime)
            _emit_protocol(
                {
                    **response,
                    "type": "response",
                    "request_id": request_id,
                    "status": "completed",
                },
                output_stream,
            )
        except RuntimeConfigurationError as error:
            _emit_protocol(
                {
                    "type": "response",
                    "request_id": request_id,
                    "status": "failed",
                    "retryable": False,
                    "error_class": type(error).__name__,
                    "error": _error_detail(error),
                },
                output_stream,
            )
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            _emit_protocol(
                {
                    "type": "response",
                    "request_id": request_id,
                    "status": "failed",
                    "retryable": True,
                    "error_class": type(error).__name__,
                    "error": _error_detail(error),
                },
                output_stream,
            )
    return 0


def main() -> int:
    if sys.argv[1:] == ["--serve"]:
        return _serve()
    if len(sys.argv) != 3:
        print(
            "usage: indicf5_runtime.py --serve | REQUEST RESPONSE",
            file=sys.stderr,
        )
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

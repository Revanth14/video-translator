from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from dub_mvp.indicf5_runtime import (
    RuntimeConfigurationError,
    _positive_float,
    _read_request,
    _serve,
    _validate_batches,
)


def valid_request(**overrides: object) -> dict[str, object]:
    request = {
        "schema_version": 5,
        "model": "ai4bharat/IndicF5",
        "model_revision": "ai4bharat_indicf5_v1",
        "translated_text": "नमस्ते दुनिया।",
        "tts_text": "नमस्ते दुनिया।",
        "text_normalization_policy": "hindi_codeswitch_v1",
        "text_batches": ["नमस्ते दुनिया।"],
        "output_path": "/tmp/out.wav",
        "reference_audio": "/tmp/ref.wav",
        "reference_text": "मेरा नाम राहुल है।",
        "reference_seconds": 9.0,
        "fix_duration_seconds": 13.5,
    }
    request.update(overrides)
    return request


def test_read_request_requires_the_duration_fields(tmp_path: Path) -> None:
    legacy = valid_request()
    del legacy["fix_duration_seconds"]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError, match="missing required fields"):
        _read_request(path)


def test_read_request_accepts_a_complete_request(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(valid_request()), encoding="utf-8")

    assert _read_request(path)["fix_duration_seconds"] == 13.5


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "model_revision",
        "translated_text",
        "tts_text",
        "text_normalization_policy",
    ],
)
def test_read_request_rejects_empty_text_contract_fields(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(valid_request(**{field: "  "})), encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError, match=field):
        _read_request(path)


def test_read_request_rejects_the_previous_text_schema(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(valid_request(schema_version=3, target_text="नमस्ते दुनिया।")),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigurationError, match="expected version 5"):
        _read_request(path)


def test_validate_batches_rejects_multiple_batches() -> None:
    # fix_duration is applied per batch, so more than one batch would pin each
    # fragment to the whole utterance's window.
    request = valid_request(
        text_batches=["नमस्ते", "दुनिया।"],
        tts_text="नमस्ते दुनिया।",
    )

    with pytest.raises(RuntimeConfigurationError, match="exactly one text batch"):
        _validate_batches(request)


def test_validate_batches_accepts_one_batch() -> None:
    assert _validate_batches(valid_request()) == ["नमस्ते दुनिया।"]


def test_validate_batches_rejects_altered_text() -> None:
    request = valid_request(text_batches=["नमस्ते"])

    with pytest.raises(RuntimeConfigurationError, match="alter the TTS text"):
        _validate_batches(request)


@pytest.mark.parametrize("value", [0, -1.0, "abc", None, float("inf"), float("nan")])
def test_positive_float_rejects_unusable_values(value: object) -> None:
    with pytest.raises(RuntimeConfigurationError):
        _positive_float({"fix_duration_seconds": value}, "fix_duration_seconds")


def test_positive_float_accepts_a_measurement() -> None:
    assert _positive_float({"reference_seconds": "9.5"}, "reference_seconds") == 9.5


def test_server_loads_runtime_once_and_correlates_sequential_requests() -> None:
    loaded: list[object] = []
    synthesized: list[str] = []
    runtime = object()

    def load_runtime() -> object:
        loaded.append(runtime)
        return runtime

    def synthesize(request, *, runtime):
        assert runtime is loaded[0]
        synthesized.append(request["request_id"])
        return {"duration_ms": 1000, "seed": None}

    first = valid_request(request_id="request-1")
    second = valid_request(request_id="request-2")
    output = StringIO()

    exit_code = _serve(
        input_stream=StringIO(
            json.dumps(first) + "\n" + json.dumps(second) + "\n"
        ),
        output_stream=output,
        runtime_loader=load_runtime,
        synthesizer=synthesize,
    )

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert exit_code == 0
    assert loaded == [runtime]
    assert synthesized == ["request-1", "request-2"]
    assert messages[0]["status"] == "ready"
    assert [message["request_id"] for message in messages[1:]] == [
        "request-1",
        "request-2",
    ]
    assert all(message["status"] == "completed" for message in messages[1:])


def test_server_rejects_reused_request_id_without_reloading_runtime() -> None:
    load_count = 0

    def load_runtime() -> object:
        nonlocal load_count
        load_count += 1
        return object()

    request = valid_request(request_id="duplicate")
    output = StringIO()

    assert _serve(
        input_stream=StringIO(json.dumps(request) + "\n" + json.dumps(request)),
        output_stream=output,
        runtime_loader=load_runtime,
        synthesizer=lambda request, *, runtime: {"duration_ms": 1000},
    ) == 0

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert load_count == 1
    assert messages[1]["status"] == "completed"
    assert messages[2]["status"] == "failed"
    assert messages[2]["request_id"] == "duplicate"
    assert messages[2]["retryable"] is False


def test_server_reports_configuration_failure_as_permanent() -> None:
    output = StringIO()

    def fail_to_load() -> object:
        raise RuntimeConfigurationError("missing model")

    assert _serve(
        input_stream=StringIO(),
        output_stream=output,
        runtime_loader=fail_to_load,
    ) == 2

    ready = json.loads(output.getvalue())
    assert ready["type"] == "ready"
    assert ready["status"] == "failed"
    assert ready["retryable"] is False
    assert ready["error"] == "missing model"

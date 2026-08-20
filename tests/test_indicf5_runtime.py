from __future__ import annotations

import json
from pathlib import Path

import pytest

from dub_mvp.indicf5_runtime import (
    RuntimeConfigurationError,
    _positive_float,
    _read_request,
    _validate_batches,
)


def valid_request(**overrides: object) -> dict[str, object]:
    request = {
        "schema_version": 3,
        "target_text": "नमस्ते दुनिया।",
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


def test_read_request_rejects_the_legacy_byte_budget_schema(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(valid_request(schema_version=2, max_chunk_bytes=200)),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigurationError, match="expected version 3"):
        _read_request(path)


def test_validate_batches_rejects_multiple_batches() -> None:
    # fix_duration is applied per batch, so more than one batch would pin each
    # fragment to the whole utterance's window.
    request = valid_request(
        text_batches=["नमस्ते", "दुनिया।"],
        target_text="नमस्ते दुनिया।",
    )

    with pytest.raises(RuntimeConfigurationError, match="exactly one text batch"):
        _validate_batches(request)


def test_validate_batches_accepts_one_batch() -> None:
    assert _validate_batches(valid_request()) == ["नमस्ते दुनिया।"]


def test_validate_batches_rejects_altered_text() -> None:
    request = valid_request(text_batches=["नमस्ते"])

    with pytest.raises(RuntimeConfigurationError, match="alter the target text"):
        _validate_batches(request)


@pytest.mark.parametrize("value", [0, -1.0, "abc", None, float("inf"), float("nan")])
def test_positive_float_rejects_unusable_values(value: object) -> None:
    with pytest.raises(RuntimeConfigurationError):
        _positive_float({"fix_duration_seconds": value}, "fix_duration_seconds")


def test_positive_float_accepts_a_measurement() -> None:
    assert _positive_float({"reference_seconds": "9.5"}, "reference_seconds") == 9.5

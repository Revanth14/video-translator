import shutil
from datetime import datetime
from pathlib import Path

import pytest

from dub_mvp.artifacts import (
    completed_artifact_metadata,
    fingerprint_inputs,
    verify_artifact,
    write_artifact_metadata,
)


def test_input_fingerprint_is_canonical() -> None:
    first = fingerprint_inputs({"voice": "a", "text": "hello"})
    second = fingerprint_inputs({"text": "hello", "voice": "a"})

    assert first == second
    assert first != fingerprint_inputs({"text": "hello", "voice": "b"})


def test_input_fingerprint_rejects_timestamps() -> None:
    with pytest.raises(TypeError, match="Timestamps cannot take part"):
        fingerprint_inputs({"created_at": datetime(2026, 8, 12)})


def test_completed_artifact_verifies_inputs_and_checksum(tmp_path: Path) -> None:
    audio = tmp_path / "speech" / "utterance.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"valid audio")
    inputs = {"text": "namaste", "voice": "voice-a"}
    metadata = completed_artifact_metadata(
        artifact_id="utt_0001-r1",
        kind="tts_audio",
        path=audio,
        root=tmp_path,
        inputs=inputs,
        provider="fixture",
        model="fixture-tts",
    )

    assert metadata.path == "speech/utterance.wav"
    assert verify_artifact(metadata, expected_inputs=inputs, root=tmp_path).valid
    assert not verify_artifact(
        metadata,
        expected_inputs={"text": "namaste", "voice": "voice-b"},
        root=tmp_path,
    ).valid

    audio.write_bytes(b"corrupt audio")
    result = verify_artifact(metadata, expected_inputs=inputs, root=tmp_path)
    assert not result.valid
    assert result.reason in {"artifact size mismatch", "artifact checksum mismatch"}


def test_artifact_stays_verifiable_after_the_run_moves(tmp_path: Path) -> None:
    original_root = tmp_path / "runs" / "run-a"
    audio = original_root / "speech" / "utterance.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"valid audio")
    inputs = {"text": "namaste"}
    metadata = completed_artifact_metadata(
        artifact_id="utt_0001-r1",
        kind="tts_audio",
        path=audio,
        root=original_root,
        inputs=inputs,
    )

    moved_root = tmp_path / "archive" / "run-a"
    moved_root.parent.mkdir()
    shutil.move(str(original_root), str(moved_root))

    assert verify_artifact(metadata, expected_inputs=inputs, root=moved_root).valid


def test_artifact_outside_the_run_directory_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"valid audio")
    root = tmp_path / "run-a"
    root.mkdir()

    with pytest.raises(ValueError, match="outside its run directory"):
        completed_artifact_metadata(
            artifact_id="utt_0001-r1",
            kind="tts_audio",
            path=outside,
            root=root,
            inputs={"text": "namaste"},
        )


def test_artifact_sidecar_write_is_atomic(tmp_path: Path) -> None:
    audio = tmp_path / "utterance.wav"
    audio.write_bytes(b"valid audio")
    metadata = completed_artifact_metadata(
        artifact_id="utt_0001-r1",
        kind="tts_audio",
        path=audio,
        root=tmp_path,
        inputs={"text": "namaste"},
    )
    sidecar = tmp_path / "utterance.json"

    assert write_artifact_metadata(sidecar, metadata) == sidecar
    assert sidecar.is_file()
    assert not (tmp_path / ".utterance.json.tmp").exists()


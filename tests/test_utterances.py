import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from dub_mvp.artifacts import (
    ArtifactMetadata,
    fingerprint_inputs,
    sha256_file,
    verify_artifact,
)
from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.transcribe import (
    NormalizedTranscript,
    TranscriptSegment,
    TranscriptUtterance,
    TranscriptWord,
)
from dub_mvp.utterances import (
    DubbingUtterance,
    DubbingUtteranceArtifact,
    OverlapStatus,
    UtterancePipeline,
    build_dubbing_utterances,
)


def utterance(
    utterance_id: str,
    *,
    start_ms: int,
    end_ms: int,
    overlap_status: OverlapStatus = OverlapStatus.NONE,
) -> DubbingUtterance:
    return DubbingUtterance(
        utterance_id=utterance_id,
        speaker_id="speaker_01",
        start_ms=start_ms,
        end_ms=end_ms,
        available_duration_ms=end_ms - start_ms,
        source_text="Hello world.",
        source_segment_ids=["seg_0001"],
        source_word_indexes=[0, 1],
        overlap_status=overlap_status,
        confidence=0.95,
    )


def test_dubbing_utterance_validates_duration_and_references() -> None:
    item = utterance("utt_0001", start_ms=100, end_ms=1100)

    assert item.available_duration_ms == 1000
    assert item.speaker_id == "speaker_01"


def test_dubbing_utterance_rejects_incorrect_duration() -> None:
    with pytest.raises(ValidationError, match="Available duration"):
        DubbingUtterance(
            utterance_id="utt_0001",
            start_ms=100,
            end_ms=1100,
            available_duration_ms=999,
            source_text="Hello.",
        )


def test_utterance_artifact_requires_explicit_overlap_status() -> None:
    fingerprint = fingerprint_inputs({"pause_split_ms": 700})

    with pytest.raises(ValidationError, match="marked explicitly"):
        DubbingUtteranceArtifact(
            source_language="en",
            source_transcript="metadata/transcript.json",
            configuration_fingerprint=fingerprint,
            utterances=[
                utterance("utt_0001", start_ms=0, end_ms=1000),
                utterance("utt_0002", start_ms=900, end_ms=1500),
            ],
        )

    artifact = DubbingUtteranceArtifact(
        source_language="en",
        source_transcript="metadata/transcript.json",
        configuration_fingerprint=fingerprint,
        utterances=[
            utterance(
                "utt_0001",
                start_ms=0,
                end_ms=1000,
                overlap_status=OverlapStatus.CONFIRMED,
            ),
            utterance(
                "utt_0002",
                start_ms=900,
                end_ms=1500,
                overlap_status=OverlapStatus.CONFIRMED,
            ),
        ],
    )

    assert len(artifact.utterances) == 2


def test_utterance_artifact_detects_overlap_with_a_containing_utterance() -> None:
    fingerprint = fingerprint_inputs({"pause_split_ms": 700})

    # utt_0003 sits inside utt_0001 but does not touch utt_0002, so a pairwise
    # comparison against only the previous utterance would miss it.
    with pytest.raises(ValidationError, match="marked explicitly"):
        DubbingUtteranceArtifact(
            source_language="en",
            source_transcript="metadata/transcript.json",
            configuration_fingerprint=fingerprint,
            utterances=[
                utterance("utt_0001", start_ms=0, end_ms=10000),
                utterance(
                    "utt_0002",
                    start_ms=1000,
                    end_ms=2000,
                    overlap_status=OverlapStatus.CONFIRMED,
                ),
                utterance("utt_0003", start_ms=3000, end_ms=4000),
            ],
        )


def speaker_transcript() -> NormalizedTranscript:
    words = [
        TranscriptWord(
            text="Hello",
            start_ms=100,
            end_ms=500,
            confidence=0.9,
            speaker_id="speaker_a",
        ),
        TranscriptWord(
            text="there.",
            start_ms=500,
            end_ms=900,
            confidence=0.7,
            speaker_id="speaker_a",
        ),
        TranscriptWord(
            text="General",
            start_ms=1200,
            end_ms=1600,
            confidence=0.8,
            speaker_id="speaker_b",
        ),
        TranscriptWord(
            text="Kenobi.",
            start_ms=1600,
            end_ms=2100,
            speaker_id="speaker_b",
        ),
    ]
    return NormalizedTranscript(
        model="test-model",
        language="en",
        duration_ms=3000,
        utterances=[
            TranscriptUtterance(
                utterance_id="provider_0001",
                start_ms=100,
                end_ms=2100,
                text="Hello there. General Kenobi.",
                words=words,
            )
        ],
    )


def source_segment() -> TranscriptSegment:
    return TranscriptSegment(
        segment_id="seg_0001",
        start_ms=100,
        end_ms=2100,
        duration_budget_ms=2000,
        source_text="Hello there. General Kenobi.",
        word_indexes=[0, 1, 2, 3],
    )


def test_build_dubbing_utterances_splits_speakers_with_stable_traceability() -> None:
    items = build_dubbing_utterances(speaker_transcript(), [source_segment()])

    assert [item.utterance_id for item in items] == ["utt_0001", "utt_0002"]
    assert [item.speaker_id for item in items] == ["speaker_a", "speaker_b"]
    assert items[0].source_word_indexes == [0, 1]
    assert items[1].source_word_indexes == [2, 3]
    assert items[0].source_segment_ids == ["seg_0001"]
    assert items[0].following_context == "General Kenobi."
    assert items[1].preceding_context == "Hello there."
    assert items[0].confidence == pytest.approx(0.8)


def test_utterance_pipeline_writes_translation_contract_and_metadata(
    tmp_path: Path,
) -> None:
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    transcript_path = metadata_directory / "transcript.json"
    segments_path = metadata_directory / "segments.json"
    transcript_path.write_text(
        json.dumps(speaker_transcript().model_dump(mode="json")),
        encoding="utf-8",
    )
    segments_path.write_text(
        json.dumps([source_segment().model_dump(mode="json")]),
        encoding="utf-8",
    )

    artifact, translation_segments, outputs = UtterancePipeline().run(
        transcript_path=transcript_path,
        segments_path=segments_path,
        run_directory=tmp_path,
    )

    assert artifact.source_transcript == "metadata/transcript.json"
    assert [item.segment_id for item in translation_segments] == [
        "utt_0001",
        "utt_0002",
    ]
    assert all(Path(path).is_file() for path in outputs.values())
    metadata = ArtifactMetadata.model_validate_json(
        Path(outputs["dubbing_utterances_metadata"]).read_text(
            encoding="utf-8"
        )
    )
    inputs = {
        "algorithm": "speaker_turn_v1",
        "transcript_sha256": sha256_file(transcript_path),
        "segments_sha256": sha256_file(segments_path),
    }
    assert verify_artifact(
        metadata,
        expected_inputs=inputs,
        root=tmp_path,
    ).valid


def test_segment_command_persists_stage_and_resumes(tmp_path: Path) -> None:
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    transcript_path = metadata_directory / "transcript.json"
    segments_path = metadata_directory / "segments.json"
    transcript_path.write_text(
        json.dumps(speaker_transcript().model_dump(mode="json")),
        encoding="utf-8",
    )
    segments_path.write_text(
        json.dumps([source_segment().model_dump(mode="json")]),
        encoding="utf-8",
    )
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=3000,
    )
    manifest.outputs.update(
        {"transcript": str(transcript_path), "segments": str(segments_path)}
    )
    manifest.stages["transcribe"].status = StageStatus.COMPLETED
    manifest.save(tmp_path)

    runner = CliRunner()
    first = runner.invoke(app, ["segment", str(tmp_path)])
    second = runner.invoke(app, ["segment", str(tmp_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already complete" in second.output
    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.SEGMENTED
    assert loaded.stages["segment"].status == StageStatus.COMPLETED
    assert Path(loaded.outputs["translation_segments"]).is_file()

import pytest
from pydantic import ValidationError

from dub_mvp.artifacts import fingerprint_inputs
from dub_mvp.utterances import (
    DubbingUtterance,
    DubbingUtteranceArtifact,
    OverlapStatus,
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

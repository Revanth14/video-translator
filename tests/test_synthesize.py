import json
import os
import sys
import wave
from multiprocessing import Process
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dub_mvp.artifacts import ArtifactMetadata
from dub_mvp.cli import app
from dub_mvp.duration import DurationCorrectionError, DurationCorrector, DurationPolicy
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.synthesize import (
    SpeechAttempt,
    SpeechAttemptStatus,
    SpeechProviderError,
    IndicF5Provider,
    SynthesisError,
    SynthesisMetrics,
    SynthesisPipeline,
    SynthesisResult,
    VoiceCatalog,
    VoiceReference,
    load_localized_segments,
    load_voice_catalog,
    load_voice_reference,
    synthesis_outputs_reusable,
)


FIXTURES = Path(__file__).parent / "fixtures"
LOCALIZED = FIXTURES / "localized_segments_smoke.json"
VOICE_REFERENCE = FIXTURES / "voice_reference_smoke.json"


class FixtureSpeechProvider:
    def __init__(
        self,
        *,
        write_audio: bool = True,
        provider_name: str = "fixture-tts",
        model_name: str = "fixture-indicf5",
        fail_on: str | None = None,
    ) -> None:
        self.write_audio = write_audio
        self.provider_name = provider_name
        self.model_name = model_name
        self.fail_on = fail_on
        self.calls: list[dict[str, Any]] = []

    def synthesize(
        self,
        segment,
        *,
        output_path: Path,
        voice_reference,
        target_language: str,
        revision: int,
    ) -> SynthesisResult:
        self.calls.append(
            {
                "segment_id": segment.segment_id,
                "reference_id": voice_reference.reference_id,
                "target_language": target_language,
                "revision": revision,
            }
        )
        if segment.segment_id == self.fail_on:
            raise SpeechProviderError(f"failed {segment.segment_id}")
        if self.write_audio:
            write_wav(output_path, segment.duration_budget_ms - 100)
        return SynthesisResult(
            audio_path=str(output_path),
            duration_ms=segment.duration_budget_ms - 100,
            seed=revision,
            notes=["fixture"],
        )


def write_wav(path: Path, duration_ms: int, *, frame_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = duration_ms * frame_rate // 1000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_indicf5_provider_runs_isolated_runtime_with_balanced_batches(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "fake_indicf5_runtime.py"
    runtime.write_text(
        """
import json
import sys
import wave
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[1]).with_name("captured-request.json").write_text(json.dumps(request))
output = Path(request["output_path"])
output.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(output), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(24000)
    handle.writeframes(b"\\x00\\x00" * 24000)
Path(sys.argv[2]).write_text(json.dumps({"duration_ms": 3200, "seed": 42}))
""",
        encoding="utf-8",
    )
    reference_audio = tmp_path / "reference.wav"
    write_wav(reference_audio, 9_000)
    provider = IndicF5Provider(
        runtime_python=sys.executable,
        runtime_script=runtime,
    )
    segment = load_localized_segments(LOCALIZED)[0]
    voice_reference = VoiceReference(
        reference_id="approved-reference",
        path=str(reference_audio),
        reference_text="This is an exact and sufficiently detailed reference transcript.",
        consent="approved fixture",
    )

    result = provider.synthesize(
        segment,
        output_path=tmp_path / "tts-r1.wav",
        voice_reference=voice_reference,
        target_language="hi",
        revision=1,
    )

    assert result.duration_ms == 3200
    assert result.seed == 42
    assert "indicf5_chunk_policy=single_batch_v1" in result.notes
    assert "indicf5_duration_policy=fixed_timeline_budget_v1" in result.notes
    assert not list(tmp_path.glob("*.indicf5-*.json"))

    request = json.loads((tmp_path / "captured-request.json").read_text())
    # Generation is pinned to reference + timeline budget rather than left to
    # the UTF-8 byte ratio that mistimed the evaluation samples.
    assert request["target_duration_ms"] == segment.duration_budget_ms
    assert request["reference_seconds"] == pytest.approx(9.0)
    assert request["fix_duration_seconds"] == pytest.approx(12.28)
    assert request["schema_version"] == 3
    assert "max_chunk_bytes" not in request


def test_indicf5_provider_rejects_over_long_reference_audio(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    write_wav(reference_audio, 15_000)
    provider = IndicF5Provider(
        runtime_python=sys.executable,
        runtime_script=tmp_path / "unused.py",
    )
    segment = load_localized_segments(LOCALIZED)[0]
    voice_reference = VoiceReference(
        reference_id="approved-reference",
        path=str(reference_audio),
        reference_text="This is an exact and sufficiently detailed reference transcript.",
        consent="approved fixture",
    )

    with pytest.raises(SynthesisError, match="clips anything over"):
        provider.synthesize(
            segment,
            output_path=tmp_path / "tts-r1.wav",
            voice_reference=voice_reference,
            target_language="hi",
            revision=1,
        )


def test_indicf5_provider_records_cross_script_prompting(tmp_path: Path) -> None:
    """Source-clone dubbing prompts Hindi with the speaker's English audio.

    That pairing is the product, so it must proceed and be recorded rather than
    rejected. Generation is pinned with fix_duration, so the byte ratio that
    once made a cross-script reference harmful never runs.
    """
    runtime = tmp_path / "fake_indicf5_runtime.py"
    runtime.write_text(
        """
import json
import sys
import wave
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
output = Path(request["output_path"])
output.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(output), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(24000)
    handle.writeframes(b"\\x00\\x00" * 24000)
Path(sys.argv[2]).write_text(json.dumps({"duration_ms": 3280}))
""",
        encoding="utf-8",
    )
    reference_audio = tmp_path / "reference.wav"
    write_wav(reference_audio, 9_000)
    provider = IndicF5Provider(
        runtime_python=sys.executable,
        runtime_script=runtime,
    )
    # This is the 306-byte representative long case. The retired
    # reference-derived byte budget admitted only 190 bytes and stopped this
    # request before the GPU despite its duration fitting the 25-second window.
    segment = load_localized_segments(LOCALIZED)[0].model_copy(
        update={
            "end_ms": 11_120,
            "duration_budget_ms": 11_000,
            "target_text": (
                "आज हम सीखेंगे कि किसी भी वीडियो को एक भाषा से दूसरी भाषा में "
                "कैसे बदला जाता है, और इसमें आवाज़ को कैसे बनाए रखा जाता है।"
            ),
        }
    )
    voice_reference = VoiceReference(
        reference_id="approved-reference",
        path=str(reference_audio),
        reference_text="My email is one at the rate gmail.com and it works.",
        consent="approved fixture",
    )

    result = provider.synthesize(
        segment,
        output_path=tmp_path / "tts-r1.wav",
        voice_reference=voice_reference,
        target_language="hi",
        revision=1,
    )

    assert "indicf5_scripts_match=false" in result.notes
    assert "indicf5_reference_script=latin" in result.notes
    assert "indicf5_target_script=devanagari" in result.notes
    # Incomparable unit counts, so no scale is claimed.
    assert not any(
        note.startswith("indicf5_implied_rate_scale=") for note in result.notes
    )


def test_indicf5_provider_requires_exact_reference_transcript(
    tmp_path: Path,
) -> None:
    reference_audio = tmp_path / "reference.wav"
    write_wav(reference_audio, 10_000)
    provider = IndicF5Provider(
        runtime_python=sys.executable,
        runtime_script=tmp_path / "unused.py",
    )
    segment = load_localized_segments(LOCALIZED)[0]
    voice_reference = VoiceReference(
        reference_id="approved-reference",
        path=str(reference_audio),
        consent="approved fixture",
    )

    with pytest.raises(SynthesisError, match="exact transcript"):
        provider.synthesize(
            segment,
            output_path=tmp_path / "tts-r1.wav",
            voice_reference=voice_reference,
            target_language="hi",
            revision=1,
        )


def test_loads_localized_segments_and_voice_reference() -> None:
    segments = load_localized_segments(LOCALIZED)
    voice_reference = load_voice_reference(VOICE_REFERENCE)

    assert [segment.segment_id for segment in segments] == [
        "seg_0001",
        "seg_0002",
    ]
    assert voice_reference.reference_id == "generic-hindi-fixture"
    assert load_voice_catalog(VOICE_REFERENCE).voices == [voice_reference]


def test_legacy_synthesized_segment_schema_is_migrated() -> None:
    from dub_mvp.render import load_synthesized_segments

    segments = load_synthesized_segments(
        FIXTURES / "synthesized_segments_smoke.json"
    )

    assert {segment.schema_version for segment in segments} == {1}


def test_phase_nine_synthesized_segment_schema_remains_renderable(
    tmp_path: Path,
) -> None:
    from dub_mvp.render import load_synthesized_segments

    payload = json.loads(
        (FIXTURES / "synthesized_segments_smoke.json").read_text(encoding="utf-8")
    )
    payload[0]["schema_version"] = 2
    path = tmp_path / "phase-nine-synthesized.json"
    path.write_text(json.dumps(payload[:1]), encoding="utf-8")

    loaded = load_synthesized_segments(path)

    assert loaded[0].schema_version == 2
    assert loaded[0].duration_status == "legacy_unfitted"


def test_pipeline_writes_revisioned_audio_and_metadata(tmp_path: Path) -> None:
    localized_path = tmp_path / "localized_segments.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    provider = FixtureSpeechProvider()

    synthesized, outputs, model_name = SynthesisPipeline(
        provider=provider,
    ).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert model_name == "fixture-indicf5"
    assert [call["revision"] for call in provider.calls] == [1, 1]
    assert [segment.tts_revision for segment in synthesized] == [1, 1]
    assert Path(synthesized[0].tts_audio_path).is_file()
    assert synthesized[0].tts_duration_ms == 3180
    assert Path(outputs["synthesis_raw"]).is_file()
    assert Path(outputs["synthesis_raw_metadata"]).is_file()
    assert Path(outputs["synthesized_segments"]).is_file()
    assert Path(outputs["synthesized_segments_metadata"]).is_file()
    assert Path(outputs["speaker_voice_map"]).is_file()
    assert Path(outputs["synthesis_metrics_metadata"]).is_file()
    voice_map_metadata = ArtifactMetadata.model_validate_json(
        Path(outputs["speaker_voice_map_metadata"]).read_text(encoding="utf-8")
    )
    assert not Path(voice_map_metadata.path).is_absolute()
    metrics = SynthesisMetrics.model_validate_json(
        Path(outputs["synthesis_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics.provider_calls == 2
    assert metrics.generated_duration_ms == 7480


def test_pipeline_uses_next_revision_without_overwriting(
    tmp_path: Path,
) -> None:
    localized_path = tmp_path / "localized_segments.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    provider = FixtureSpeechProvider()

    first, _, _ = SynthesisPipeline(provider=provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )
    existing = Path(first[0].tts_audio_path)
    accepted = existing.read_bytes()
    synthesized, _, _ = SynthesisPipeline(provider=provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
        reuse_completed_utterances=False,
    )

    assert synthesized[0].tts_revision == 2
    assert Path(synthesized[0].tts_audio_path).name.endswith("r0002.wav")
    assert existing.read_bytes() == accepted


def test_pipeline_rejects_missing_audio_output(tmp_path: Path) -> None:
    localized_path = tmp_path / "localized_segments.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))

    try:
        SynthesisPipeline(
            provider=FixtureSpeechProvider(write_audio=False),
        ).run(
            localized_segments_path=localized_path,
            run_directory=tmp_path,
            target_language="hi",
            voice_reference_path=VOICE_REFERENCE,
        )
    except SynthesisError as error:
        assert "missing or empty audio" in str(error)
    else:
        raise AssertionError("Expected missing audio output to fail.")


def test_pipeline_rejects_corrupt_provider_audio_before_completion(
    tmp_path: Path,
) -> None:
    class CorruptProvider(FixtureSpeechProvider):
        def synthesize(self, segment, *, output_path: Path, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"not a wav")
            return SynthesisResult(audio_path=str(output_path), duration_ms=10)

    localized = json.loads(LOCALIZED.read_text(encoding="utf-8"))[:1]
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(json.dumps(localized), encoding="utf-8")

    try:
        SynthesisPipeline(provider=CorruptProvider()).run(
            localized_segments_path=localized_path,
            run_directory=tmp_path,
            target_language="hi",
            voice_reference_path=VOICE_REFERENCE,
        )
    except SpeechProviderError as error:
        assert "decode generated WAV" in str(error)
    else:
        raise AssertionError("Expected corrupt provider audio to fail.")

    assert not list((tmp_path / "speech").glob("**/*.result.meta.json"))


def test_pipeline_persists_deterministic_speaker_voice_map(
    tmp_path: Path,
) -> None:
    localized = json.loads(LOCALIZED.read_text(encoding="utf-8"))
    localized[0]["speaker_id"] = "speaker_01"
    localized[1]["speaker_id"] = "speaker_02"
    localized.append(
        {
            **localized[0],
            "segment_id": "seg_0003",
            "start_ms": 9000,
            "end_ms": 10000,
            "duration_budget_ms": 1000,
        }
    )
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(json.dumps(localized), encoding="utf-8")
    catalog_path = tmp_path / "voices.json"
    catalog_path.write_text(
        VoiceCatalog(
            voices=[
                {
                    "reference_id": "stock_voice_A",
                    "path": None,
                    "consent": "stock voice",
                },
                {
                    "reference_id": "stock_voice_B",
                    "path": None,
                    "consent": "stock voice",
                },
            ]
        ).model_dump_json(),
        encoding="utf-8",
    )
    class MapAwareProvider(FixtureSpeechProvider):
        def synthesize(self, *args, **kwargs):
            assert list(
                (tmp_path / "speech" / "voice-maps").glob("*.meta.json")
            )
            return super().synthesize(*args, **kwargs)

    provider = MapAwareProvider()

    _, outputs, _ = SynthesisPipeline(provider=provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=catalog_path,
    )

    voice_map = json.loads(
        Path(outputs["speaker_voice_map"]).read_text(encoding="utf-8")
    )
    assert voice_map["assignments"] == [
        {"speaker_id": "speaker_01", "reference_id": "stock_voice_A"},
        {"speaker_id": "speaker_02", "reference_id": "stock_voice_B"},
    ]
    assert [call["reference_id"] for call in provider.calls] == [
        "stock_voice_A",
        "stock_voice_B",
        "stock_voice_A",
    ]


def test_pipeline_reuses_verified_utterances_without_provider_calls(
    tmp_path: Path,
) -> None:
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    first_provider = FixtureSpeechProvider()
    _, first_outputs, _ = SynthesisPipeline(provider=first_provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )
    resumed_provider = FixtureSpeechProvider()

    _, resumed_outputs, _ = SynthesisPipeline(provider=resumed_provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert resumed_provider.calls == []
    metrics = SynthesisMetrics.model_validate_json(
        Path(resumed_outputs["synthesis_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics.reused_utterances == 2
    assert metrics.provider_calls == 0
    assert synthesis_outputs_reusable(
        outputs=resumed_outputs,
        localized_segments_path=localized_path,
        voice_reference_path=VOICE_REFERENCE,
        run_directory=tmp_path,
        target_language="hi",
        provider_name="fixture-tts",
        model_name="fixture-indicf5",
    )
    assert Path(first_outputs["speaker_voice_map"]) == Path(
        resumed_outputs["speaker_voice_map"]
    )
    duration_histories = list(
        (tmp_path / "speech" / "duration").glob("**/*.attempts.json")
    )
    assert all(
        len(json.loads(path.read_text(encoding="utf-8"))) == 1
        for path in duration_histories
    )


def test_pipeline_rebuilds_corrupt_duration_fit_without_repeating_raw_tts(
    tmp_path: Path,
) -> None:
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    _, outputs, _ = SynthesisPipeline(
        provider=FixtureSpeechProvider()
    ).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )
    synthesized = json.loads(
        Path(outputs["synthesized_segments"]).read_text(encoding="utf-8")
    )
    fit_path = tmp_path / synthesized[0]["duration_correction_path"]
    fit_path.write_text("corrupt", encoding="utf-8")
    assert not synthesis_outputs_reusable(
        outputs=outputs,
        localized_segments_path=localized_path,
        voice_reference_path=VOICE_REFERENCE,
        run_directory=tmp_path,
        target_language="hi",
        provider_name="fixture-tts",
        model_name="fixture-indicf5",
    )
    resumed = FixtureSpeechProvider()

    regenerated, _, _ = SynthesisPipeline(provider=resumed).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert resumed.calls == []
    assert Path(regenerated[0].duration_correction_path or "").name.endswith(
        "r0002.result.json"
    )


def test_duration_policy_change_reuses_raw_speech_but_refits_timing(
    tmp_path: Path,
) -> None:
    class UnavailableTransformer:
        name = "policy-test-transformer"

        def trim_artificial_pauses(self, *args, **kwargs):
            raise DurationCorrectionError("unavailable")

        def time_stretch(self, *args, **kwargs):
            raise DurationCorrectionError("unavailable")

    localized_path = tmp_path / "localized.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    first_provider = FixtureSpeechProvider()
    first, _, _ = SynthesisPipeline(provider=first_provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )
    resumed = FixtureSpeechProvider()
    strict = DurationCorrector(
        policy=DurationPolicy(
            primary_ratio_tolerance=0.01,
            primary_absolute_tolerance_ms=0,
        ),
        transformer=UnavailableTransformer(),
    )

    refitted, _, _ = SynthesisPipeline(
        provider=resumed,
        duration_corrector=strict,
    ).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert resumed.calls == []
    assert refitted[0].original_tts_audio_path == first[0].original_tts_audio_path
    assert refitted[0].duration_correction_path != first[0].duration_correction_path


def test_pipeline_retries_only_failed_utterance(tmp_path: Path) -> None:
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    failing = FixtureSpeechProvider(fail_on="seg_0002")

    try:
        SynthesisPipeline(provider=failing).run(
            localized_segments_path=localized_path,
            run_directory=tmp_path,
            target_language="hi",
            voice_reference_path=VOICE_REFERENCE,
        )
    except SpeechProviderError:
        pass
    else:
        raise AssertionError("Expected the second utterance to fail.")

    resumed = FixtureSpeechProvider()
    _, outputs, _ = SynthesisPipeline(provider=resumed).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert [call["segment_id"] for call in resumed.calls] == ["seg_0002"]
    attempt_files = sorted((tmp_path / "speech" / "utterances").glob(
        "**/*.attempts.json"
    ))
    histories = [
        [
            SpeechAttempt.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]
        for path in attempt_files
    ]
    retried = next(history for history in histories if len(history) == 2)
    assert [attempt.status for attempt in retried] == [
        SpeechAttemptStatus.FAILED,
        SpeechAttemptStatus.COMPLETED,
    ]
    metrics = SynthesisMetrics.model_validate_json(
        Path(outputs["synthesis_metrics"]).read_text(encoding="utf-8")
    )
    assert metrics.attempt_count == 3
    assert metrics.failed_attempts == 1


def test_pipeline_regenerates_only_corrupt_audio(tmp_path: Path) -> None:
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    synthesized, _, _ = SynthesisPipeline(
        provider=FixtureSpeechProvider()
    ).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )
    Path(synthesized[0].tts_audio_path).write_bytes(b"corrupt")
    resumed = FixtureSpeechProvider()

    regenerated, _, _ = SynthesisPipeline(provider=resumed).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert [call["segment_id"] for call in resumed.calls] == ["seg_0001"]
    assert regenerated[0].tts_revision == 2
    assert regenerated[1].tts_revision == 1


def test_pipeline_regenerates_only_utterance_with_stale_text(
    tmp_path: Path,
) -> None:
    localized_path = tmp_path / "localized.json"
    original = json.loads(LOCALIZED.read_text(encoding="utf-8"))
    localized_path.write_text(json.dumps(original), encoding="utf-8")
    SynthesisPipeline(provider=FixtureSpeechProvider()).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )
    original[1]["target_text"] = "Badla hua anuvaad."
    original[1]["target_text_revision"] = 2
    localized_path.write_text(json.dumps(original), encoding="utf-8")
    resumed = FixtureSpeechProvider()

    synthesized, _, _ = SynthesisPipeline(provider=resumed).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert [call["segment_id"] for call in resumed.calls] == ["seg_0002"]
    assert [segment.tts_revision for segment in synthesized] == [1, 1]


def test_pipeline_measures_wav_instead_of_trusting_provider_duration(
    tmp_path: Path,
) -> None:
    class LyingProvider(FixtureSpeechProvider):
        def synthesize(self, segment, *, output_path: Path, **kwargs):
            write_wav(output_path, 1250)
            return SynthesisResult(
                audio_path=str(output_path), duration_ms=9999
            )

    localized = json.loads(LOCALIZED.read_text(encoding="utf-8"))[:1]
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(json.dumps(localized), encoding="utf-8")

    synthesized, _, _ = SynthesisPipeline(provider=LyingProvider()).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert synthesized[0].original_tts_duration_ms == 1250
    assert synthesized[0].tts_duration_ms != 9999


class KillOnSecondProvider(FixtureSpeechProvider):
    def synthesize(self, segment, *, output_path: Path, **kwargs):
        if segment.segment_id == "seg_0002":
            os._exit(23)
        return super().synthesize(segment, output_path=output_path, **kwargs)


def run_until_process_death(
    localized_path: str, run_directory: str, voice_path: str
) -> None:
    SynthesisPipeline(provider=KillOnSecondProvider()).run(
        localized_segments_path=Path(localized_path),
        run_directory=Path(run_directory),
        target_language="hi",
        voice_reference_path=Path(voice_path),
    )


def test_pipeline_resumes_after_real_process_death(tmp_path: Path) -> None:
    localized_path = tmp_path / "localized.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    process = Process(
        target=run_until_process_death,
        args=(str(localized_path), str(tmp_path), str(VOICE_REFERENCE)),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 23
    resumed = FixtureSpeechProvider()

    SynthesisPipeline(provider=resumed).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert [call["segment_id"] for call in resumed.calls] == ["seg_0002"]


def test_synthesize_command_resumes_completed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    localized_path = tmp_path / "metadata" / "localized_segments.json"
    localized_path.parent.mkdir()
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["localized_segments"] = str(localized_path)
    manifest.stages["localize"].status = StageStatus.COMPLETED
    manifest.save(tmp_path)
    provider = FixtureSpeechProvider(
        provider_name="indicf5",
        model_name="ai4bharat/IndicF5",
    )

    class FakePipeline:
        def __init__(self, *, model_name: str, **_: object) -> None:
            assert model_name == "ai4bharat/IndicF5"

        def run(self, **kwargs):
            return SynthesisPipeline(provider=provider).run(**kwargs)

    monkeypatch.setattr("dub_mvp.cli.SynthesisPipeline", FakePipeline)
    runner = CliRunner()

    first = runner.invoke(
        app,
        [
            "synthesize",
            str(tmp_path),
            "--voice-reference",
            str(VOICE_REFERENCE),
        ],
    )
    second = runner.invoke(
        app,
        [
            "synthesize",
            str(tmp_path),
            "--voice-reference",
            str(VOICE_REFERENCE),
        ],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already complete" in second.output
    assert len(provider.calls) == 2

    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.SYNTHESIZED
    assert loaded.stages["synthesize"].status == StageStatus.COMPLETED
    assert loaded.stages["synthesize"].attempt_count == 1
    assert loaded.stages["synthesize"].attempts[0].status == StageStatus.COMPLETED
    assert loaded.stages["synthesize"].resources is not None
    assert Path(loaded.outputs["synthesized_segments"]).is_file()


def test_synthesize_command_force_creates_new_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    localized_path = tmp_path / "metadata" / "localized_segments.json"
    localized_path.parent.mkdir()
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["localized_segments"] = str(localized_path)
    manifest.save(tmp_path)
    provider = FixtureSpeechProvider(
        provider_name="indicf5",
        model_name="ai4bharat/IndicF5",
    )

    class FakePipeline:
        def __init__(self, *, model_name: str, **_: object) -> None:
            pass

        def run(self, **kwargs):
            return SynthesisPipeline(provider=provider).run(**kwargs)

    monkeypatch.setattr("dub_mvp.cli.SynthesisPipeline", FakePipeline)
    runner = CliRunner()
    command = [
        "synthesize",
        str(tmp_path),
        "--voice-reference",
        str(VOICE_REFERENCE),
    ]

    first = runner.invoke(app, command)
    second = runner.invoke(app, command + ["--force"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert [call["revision"] for call in provider.calls] == [1, 1, 2, 2]
    record = RunManifest.load(tmp_path).stages["synthesize"]
    assert record.attempt_count == 2
    assert [attempt.status for attempt in record.attempts] == [
        StageStatus.COMPLETED,
        StageStatus.COMPLETED,
    ]


def test_synthesize_command_failure_updates_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    localized_path = tmp_path / "metadata" / "localized_segments.json"
    localized_path.parent.mkdir()
    localized_path.write_text(json.dumps([]))
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=10000,
    )
    manifest.outputs["localized_segments"] = str(localized_path)
    manifest.save(tmp_path)

    class FakePipeline:
        def __init__(self, *, model_name: str, **_: object) -> None:
            pass

        def run(self, **kwargs):
            raise SynthesisError("synthesis exploded")

    monkeypatch.setattr("dub_mvp.cli.SynthesisPipeline", FakePipeline)
    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            str(tmp_path),
            "--voice-reference",
            str(VOICE_REFERENCE),
        ],
    )

    assert result.exit_code == 1
    loaded = RunManifest.load(tmp_path)
    assert loaded.status == RunStatus.FAILED
    assert loaded.stages["synthesize"].status == StageStatus.FAILED
    assert loaded.stages["synthesize"].error == "synthesis exploded"
    assert loaded.stages["synthesize"].attempt_count == 1
    assert loaded.stages["synthesize"].attempts[0].status == StageStatus.FAILED
    assert loaded.error_records[0].error_class == "SynthesisError"


def multi_speaker_segments(tmp_path: Path, speakers: list[str]) -> Path:
    payload = [
        {
            "segment_id": f"utt_{index:04d}",
            "start_ms": (index - 1) * 4000,
            "end_ms": index * 4000,
            "duration_budget_ms": 4000,
            "speaker_id": speaker,
            "source_text": f"line {index}",
            "target_text": f"pankti {index}",
            "target_text_revision": 1,
        }
        for index, speaker in enumerate(speakers, start=1)
    ]
    path = tmp_path / "localized.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def voice_catalog_file(tmp_path: Path, count: int) -> Path:
    payload = {
        "schema_version": 1,
        "voices": [
            {"reference_id": f"voice_{index}", "path": None, "consent": "stock"}
            for index in range(count)
        ],
    }
    path = tmp_path / "voices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_shared_voice_between_speakers_is_recorded(tmp_path: Path) -> None:
    # Three speakers, two voices: two people are given the same voice. A
    # listener notices immediately, so the run must not look clean.
    segments = multi_speaker_segments(tmp_path, ["spk_a", "spk_b", "spk_c"])
    voices = voice_catalog_file(tmp_path, 2)

    synthesized, outputs, _ = SynthesisPipeline(
        provider=FixtureSpeechProvider(),
    ).run(
        localized_segments_path=segments,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=voices,
    )

    metrics = SynthesisMetrics.model_validate(
        json.loads(Path(outputs["synthesis_metrics"]).read_text(encoding="utf-8"))
    )
    assert metrics.speaker_count == 3
    assert metrics.voice_count == 2
    assert metrics.voice_collision_count == 1

    shared = [
        segment
        for segment in synthesized
        if any("shared with another speaker" in note for note in segment.notes)
    ]
    assert {segment.speaker_id for segment in shared} == {"spk_a", "spk_c"}


def test_distinct_voices_report_no_collision(tmp_path: Path) -> None:
    segments = multi_speaker_segments(tmp_path, ["spk_a", "spk_b"])
    voices = voice_catalog_file(tmp_path, 2)

    synthesized, outputs, _ = SynthesisPipeline(
        provider=FixtureSpeechProvider(),
    ).run(
        localized_segments_path=segments,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=voices,
    )

    metrics = SynthesisMetrics.model_validate(
        json.loads(Path(outputs["synthesis_metrics"]).read_text(encoding="utf-8"))
    )
    assert metrics.voice_collision_count == 0
    assert not [
        note
        for segment in synthesized
        for note in segment.notes
        if "shared with another speaker" in note
    ]


def test_require_distinct_voices_fails_before_any_provider_call(
    tmp_path: Path,
) -> None:
    segments = multi_speaker_segments(tmp_path, ["spk_a", "spk_b", "spk_c"])
    voices = voice_catalog_file(tmp_path, 2)
    provider = FixtureSpeechProvider()

    with pytest.raises(SynthesisError, match="distinct voice"):
        SynthesisPipeline(
            provider=provider,
            require_distinct_voices=True,
        ).run(
            localized_segments_path=segments,
            run_directory=tmp_path,
            target_language="hi",
            voice_reference_path=voices,
        )

    # Failing fast avoids paying for speech that would be discarded.
    assert provider.calls == []

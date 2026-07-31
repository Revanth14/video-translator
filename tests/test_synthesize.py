import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.synthesize import (
    IndicF5Provider,
    SynthesisError,
    SynthesisPipeline,
    SynthesisResult,
    load_localized_segments,
    load_voice_reference,
)


FIXTURES = Path(__file__).parent / "fixtures"
LOCALIZED = FIXTURES / "localized_segments_smoke.json"
VOICE_REFERENCE = FIXTURES / "voice_reference_smoke.json"


class FixtureSpeechProvider:
    provider_name = "fixture-tts"
    model_name = "fixture-indicf5"

    def __init__(self, *, write_audio: bool = True) -> None:
        self.write_audio = write_audio
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
        if self.write_audio:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fixture wav bytes")
        return SynthesisResult(
            audio_path=str(output_path),
            duration_ms=segment.duration_budget_ms - 100,
            seed=revision,
            notes=["fixture"],
        )


class FakeIndicF5Module:
    def __init__(self, *, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"fixture wav bytes")
        return self.result


def test_indicf5_provider_extracts_seed_and_duration_from_dict_result(
    tmp_path: Path,
) -> None:
    provider = IndicF5Provider()
    provider._module = FakeIndicF5Module(
        result={"duration_ms": 3200, "seed": 42}
    )
    segment = load_localized_segments(LOCALIZED)[0]
    voice_reference = load_voice_reference(VOICE_REFERENCE)

    result = provider.synthesize(
        segment,
        output_path=tmp_path / "tts-r1.wav",
        voice_reference=voice_reference,
        target_language="hi",
        revision=1,
    )

    assert result.duration_ms == 3200
    assert result.seed == 42


def test_indicf5_provider_extracts_seed_and_duration_from_object_result(
    tmp_path: Path,
) -> None:
    provider = IndicF5Provider()
    provider._module = FakeIndicF5Module(
        result=SimpleNamespace(duration_ms=2800, seed=7)
    )
    segment = load_localized_segments(LOCALIZED)[0]
    voice_reference = load_voice_reference(VOICE_REFERENCE)

    result = provider.synthesize(
        segment,
        output_path=tmp_path / "tts-r1.wav",
        voice_reference=voice_reference,
        target_language="hi",
        revision=1,
    )

    assert result.duration_ms == 2800
    assert result.seed == 7


def test_loads_localized_segments_and_voice_reference() -> None:
    segments = load_localized_segments(LOCALIZED)
    voice_reference = load_voice_reference(VOICE_REFERENCE)

    assert [segment.segment_id for segment in segments] == [
        "seg_0001",
        "seg_0002",
    ]
    assert voice_reference.reference_id == "generic-hindi-fixture"


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
    assert Path(outputs["synthesized_segments"]).is_file()


def test_pipeline_uses_next_revision_without_overwriting(
    tmp_path: Path,
) -> None:
    localized_path = tmp_path / "localized_segments.json"
    localized_path.write_text(LOCALIZED.read_text(encoding="utf-8"))
    existing = tmp_path / "segments" / "seg_0001" / "tts-r1.wav"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"accepted previous revision")
    provider = FixtureSpeechProvider()

    synthesized, _, _ = SynthesisPipeline(provider=provider).run(
        localized_segments_path=localized_path,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=VOICE_REFERENCE,
    )

    assert synthesized[0].tts_revision == 2
    assert Path(synthesized[0].tts_audio_path).name == "tts-r2.wav"
    assert existing.read_bytes() == b"accepted previous revision"


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
        assert "audio is missing" in str(error)
    else:
        raise AssertionError("Expected missing audio output to fail.")


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
    provider = FixtureSpeechProvider()

    class FakePipeline:
        def __init__(self, *, model_name: str) -> None:
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
    provider = FixtureSpeechProvider()

    class FakePipeline:
        def __init__(self, *, model_name: str) -> None:
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
        def __init__(self, *, model_name: str) -> None:
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

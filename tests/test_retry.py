from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from dub_mvp.artifacts import (
    ArtifactMetadata,
    ArtifactStatus,
    completed_artifact_metadata,
    fingerprint_inputs,
    write_artifact_metadata,
)
from dub_mvp.cli import app
from dub_mvp.manifest import RunManifest, RunStatus, StageStatus
from dub_mvp.retry import RetryError, RetryStage, retry_run


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sidecar(
    run: Path,
    relative: str,
    *,
    kind: str,
    configuration: dict[str, object] | None = None,
) -> Path:
    artifact = run / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(f"payload:{relative}", encoding="utf-8")
    metadata_path = artifact.with_name(artifact.name + ".meta.json")
    write_artifact_metadata(
        metadata_path,
        completed_artifact_metadata(
            artifact_id=artifact.stem,
            kind=kind,
            path=artifact,
            root=run,
            inputs={"artifact": relative},
            configuration=configuration,
        ),
    )
    return metadata_path


def _metadata(path: Path) -> ArtifactMetadata:
    return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def _completed_run(tmp_path: Path) -> RunManifest:
    localized = _write_json(
        tmp_path / "translation" / "localized.json",
        [
            {
                "segment_id": "utt_0001",
                "start_ms": 0,
                "end_ms": 1000,
                "duration_budget_ms": 1000,
                "source_text": "one",
                "target_text": "ek",
            },
            {
                "segment_id": "utt_0002",
                "start_ms": 1000,
                "end_ms": 2000,
                "duration_budget_ms": 1000,
                "source_text": "two",
                "target_text": "do",
            },
        ],
    )
    synthesized = _write_json(
        tmp_path / "speech" / "synthesized.json",
        [
            {"segment_id": "utt_0001"},
            {"segment_id": "utt_0002"},
        ],
    )
    manifest = RunManifest(
        run_id="retry-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=2000,
        status=RunStatus.RENDERED,
        outputs={
            "localized_segments": str(localized),
            "synthesized_segments": str(synthesized),
            "dubbed_video": str(tmp_path / "render" / "dubbed.mp4"),
            "benchmark_json": str(tmp_path / "benchmark" / "benchmark.json"),
        },
    )
    for name, record in manifest.stages.items():
        record.status = StageStatus.COMPLETED
        record.attempt_count = 2 if name == "synthesize" else 1
    manifest.stages["synthesize"].outputs = {
        "synthesized_segments": str(synthesized)
    }
    manifest.stages["render"].outputs = {
        "dubbed_video": str(tmp_path / "render" / "dubbed.mp4")
    }
    manifest.save(tmp_path)
    return manifest


def test_selective_synthesis_retry_invalidates_only_requested_raw_work(
    tmp_path: Path,
) -> None:
    manifest = _completed_run(tmp_path)
    directories = {
        utterance: f"u-{fingerprint_inputs({'utterance_id': utterance})[:16]}"
        for utterance in ("utt_0001", "utt_0002")
    }
    raw_one = _sidecar(
        tmp_path,
        f"speech/utterances/{directories['utt_0001']}/one.wav",
        kind="speech_audio",
    )
    raw_two = _sidecar(
        tmp_path,
        f"speech/utterances/{directories['utt_0002']}/two.wav",
        kind="speech_audio",
    )
    duration_two = _sidecar(
        tmp_path,
        f"speech/duration/{directories['utt_0002']}/fit.json",
        kind="duration_fit_result",
    )
    aggregate = _sidecar(
        tmp_path,
        "speech/synthesized-proof.json",
        kind="synthesized_segments",
    )
    render = _sidecar(
        tmp_path,
        "render/dubbed.mp4",
        kind="dubbed_video",
    )
    benchmark = _sidecar(
        tmp_path,
        "benchmark/benchmark.json",
        kind="benchmark_json",
    )

    report = retry_run(
        tmp_path,
        from_stage=RetryStage.SYNTHESIZE,
        utterance_selectors=["2"],
        now=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert report.requested_utterance_ids == ["utt_0002"]
    assert report.affected_utterance_ids == ["utt_0002"]
    assert _metadata(raw_one).status == ArtifactStatus.COMPLETED
    for invalidated in (raw_two, duration_two, aggregate, render, benchmark):
        metadata = _metadata(invalidated)
        assert metadata.status == ArtifactStatus.INVALID
        assert metadata.configuration["operator_invalidation"]["request_id"] == (
            report.request_id
        )

    current = RunManifest.load(tmp_path)
    assert current.revision == manifest.revision + 2
    assert current.status == RunStatus.QUEUED
    assert current.stages["synthesize"].status == StageStatus.QUEUED
    assert current.stages["render"].status == StageStatus.PENDING
    assert current.stages["synthesize"].attempt_count == 2
    assert current.stages["synthesize"].max_attempts == 5
    assert "synthesized_segments" not in current.outputs
    assert "dubbed_video" not in current.outputs
    assert "benchmark_json" not in current.outputs
    assert Path(current.outputs["latest_retry"]).is_file()
    assert Path(current.outputs["latest_retry_metadata"]).is_file()
    assert current.stages["synthesize"].events[-1].event == (
        "operator_retry_queued"
    )


def test_localization_retry_expands_to_existing_batch_ownership(
    tmp_path: Path,
) -> None:
    manifest = _completed_run(tmp_path)
    translation_segments = _write_json(
        tmp_path / "segments" / "translation.json",
        [
            {"segment_id": "utt_0001"},
            {"segment_id": "utt_0002"},
            {"segment_id": "utt_0003"},
        ],
    )
    manifest = RunManifest.load(tmp_path)
    manifest.outputs["translation_segments"] = str(translation_segments)
    manifest.save(tmp_path)
    first_batch = _sidecar(
        tmp_path,
        "translation/batches/batch-1.json",
        kind="translation_batch",
        configuration={"owned_ids": ["utt_0001", "utt_0002"]},
    )
    second_batch = _sidecar(
        tmp_path,
        "translation/batches/batch-2.json",
        kind="translation_batch",
        configuration={"owned_ids": ["utt_0003"]},
    )
    localized = _sidecar(
        tmp_path,
        "translation/localized-proof.json",
        kind="localized_segments",
    )

    report = retry_run(
        tmp_path,
        from_stage=RetryStage.LOCALIZE,
        utterance_selectors=["utt_0002"],
    )

    assert report.affected_utterance_ids == ["utt_0002", "utt_0001"]
    assert _metadata(first_batch).status == ArtifactStatus.INVALID
    assert _metadata(second_batch).status == ArtifactStatus.COMPLETED
    assert _metadata(localized).status == ArtifactStatus.INVALID
    current = RunManifest.load(tmp_path)
    assert current.stages["localize"].status == StageStatus.QUEUED
    assert current.stages["synthesize"].status == StageStatus.PENDING
    assert current.stages["render"].status == StageStatus.PENDING


def test_retry_rejects_running_work_without_changing_state(
    tmp_path: Path,
) -> None:
    manifest = _completed_run(tmp_path)
    manifest = RunManifest.load(tmp_path)
    manifest.stages["render"].status = StageStatus.RUNNING
    manifest.stages["render"].worker_id = "worker-a"
    manifest.save(tmp_path)
    before = RunManifest.load(tmp_path)

    try:
        retry_run(
            tmp_path,
            from_stage=RetryStage.SYNTHESIZE,
            utterance_selectors=["utt_0001"],
        )
    except RetryError as error:
        assert "active work" in str(error)
    else:  # pragma: no cover - the assertion above is the behavior under test
        raise AssertionError("Retry should reject a running downstream stage")

    after = RunManifest.load(tmp_path)
    assert after.revision == before.revision
    assert after.updated_at == before.updated_at
    assert after.stages["render"].worker_id == "worker-a"


def test_retry_cli_accepts_numeric_utterance_suffix(tmp_path: Path) -> None:
    _completed_run(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "retry",
            "--run",
            str(tmp_path),
            "--utterances",
            "1,2",
            "--from",
            "synthesize",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Queued synthesize for 2 affected utterance(s)." in result.output

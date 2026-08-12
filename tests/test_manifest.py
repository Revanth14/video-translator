from pathlib import Path

import pytest

from dub_mvp.manifest import (
    ManifestConflictError,
    RunManifest,
    RunStatus,
    StageAttempt,
    StageStatus,
)


def test_manifest_round_trip_is_atomic(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="test-run",
        source_path="/tmp/source.mp4",
        source_start_ms=1000,
        source_end_ms=5000,
    )
    manifest.status = RunStatus.RUNNING
    manifest.stages["ingest"].status = StageStatus.RUNNING

    path = manifest.save(tmp_path)
    loaded = RunManifest.load(tmp_path)

    assert path == tmp_path / "manifest.json"
    assert loaded == manifest
    assert loaded.revision == 1
    assert loaded.duration_ms == 4000
    assert not (tmp_path / ".manifest.json.tmp").exists()


def test_manifest_summary_omits_internal_media_detail() -> None:
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )

    assert manifest.public_summary()["stages"] == {
        "ingest": "pending",
        "transcribe": "pending",
        "segment": "pending",
        "localize": "pending",
        "synthesize": "pending",
        "render": "pending",
    }
    assert manifest.public_summary()["source_language"] == "en"


def test_old_manifest_gains_segment_stage_without_reopening_completed_work() -> None:
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.outputs["segments"] = "/tmp/segments.json"
    manifest.stages["transcribe"].status = StageStatus.COMPLETED
    manifest.stages["localize"].status = StageStatus.COMPLETED
    payload = manifest.model_dump(mode="json")
    del payload["stages"]["segment"]

    migrated = RunManifest.model_validate(payload)

    assert migrated.stages["segment"].status == StageStatus.COMPLETED
    assert (
        migrated.outputs["translation_segments"]
        == migrated.outputs["segments"]
    )


def test_stage_record_supports_durable_attempt_and_lease_metadata() -> None:
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    record = manifest.stages["transcribe"]
    record.status = StageStatus.RUNNING
    record.attempt_count = 1
    record.worker_id = "worker-a"
    record.lease_generation = 3
    record.attempts.append(
        StageAttempt(
            attempt_number=1,
            status=StageStatus.RUNNING,
            started_at=manifest.created_at,
            worker_id="worker-a",
            lease_generation=3,
        )
    )

    assert record.max_attempts == 3
    assert record.retryable
    assert record.attempts[0].lease_generation == 3


def test_manifest_rejects_stale_revision(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.save(tmp_path)
    first_writer = RunManifest.load(tmp_path)
    stale_writer = RunManifest.load(tmp_path)

    first_writer.status = RunStatus.RUNNING
    first_writer.save(tmp_path)

    stale_writer.status = RunStatus.FAILED
    with pytest.raises(ManifestConflictError, match="expected revision 1"):
        stale_writer.save(tmp_path)

    assert RunManifest.load(tmp_path).status == RunStatus.RUNNING


def test_aborted_manifest_mutation_does_not_change_revision(
    tmp_path: Path,
) -> None:
    from dub_mvp.manifest import MutationAborted, mutate_manifest

    manifest = RunManifest(
        run_id="test-run",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.save(tmp_path)
    before = RunManifest.load(tmp_path)

    def abort(_: RunManifest) -> None:
        raise MutationAborted

    mutate_manifest(tmp_path, abort)
    after = RunManifest.load(tmp_path)

    assert after.revision == before.revision
    assert after.updated_at == before.updated_at

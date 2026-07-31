from pathlib import Path

from dub_mvp.manifest import RunManifest, RunStatus, StageStatus


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
    }

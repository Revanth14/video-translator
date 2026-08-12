from pathlib import Path

from dub_mvp.manifest import RunManifest, RunStatus
from dub_mvp.state_store import LocalManifestStateStore


def test_local_state_store_mutates_under_manifest_revision(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="run-a",
        source_path="source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.save(tmp_path)
    store = LocalManifestStateStore()

    updated = store.mutate(
        tmp_path,
        lambda current: setattr(current, "status", RunStatus.QUEUED),
    )

    assert updated.status == RunStatus.QUEUED
    assert store.load(tmp_path).revision == 2

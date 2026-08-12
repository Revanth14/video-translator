from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from dub_mvp.manifest import RunManifest, mutate_manifest


class RunStateStore(Protocol):
    """Durable state operations required by the local worker runtime."""

    def load(self, run_directory: Path) -> RunManifest:
        ...

    def mutate(
        self,
        run_directory: Path,
        apply: Callable[[RunManifest], None],
    ) -> RunManifest:
        ...


class LocalManifestStateStore:
    """POSIX-filesystem state store backed by locked manifest replacement.

    This backend is appropriate for local development and workers sharing a
    local disk. A remote implementation must supply equivalent conditional
    mutation semantics instead of relying on `flock`.
    """

    def load(self, run_directory: Path) -> RunManifest:
        return RunManifest.load(run_directory)

    def mutate(
        self,
        run_directory: Path,
        apply: Callable[[RunManifest], None],
    ) -> RunManifest:
        return mutate_manifest(run_directory, apply)


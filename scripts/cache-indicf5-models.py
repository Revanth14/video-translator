#!/usr/bin/env python3
"""Download and verify the exact IndicF5 artifacts qualified on the GPU."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


INDICF5_REPOSITORY = "ai4bharat/IndicF5"
INDICF5_REVISION = "ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
VOCODER_REPOSITORY = "charactr/vocos-mel-24khz"
VOCODER_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"

INDICF5_ARTIFACTS = {
    "model.safetensors": (
        "ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c"
    ),
    "checkpoints/vocab.txt": (
        "d3a5ff6aac12ea4fb50628a66a97b20c1e83b9e5ca356e5afae17b511cda96df"
    ),
}
VOCODER_ARTIFACTS = {
    "config.yaml": (
        "da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc"
    ),
    "pytorch_model.bin": (
        "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(root: Path, expected: dict[str, str]) -> list[str]:
    """Return explicit integrity failures for a qualified artifact set."""

    failures: list[str] = []
    for relative_path, expected_sha256 in expected.items():
        path = root / relative_path
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            failures.append(
                f"checksum mismatch for {path}: {actual_sha256}"
            )
    return failures


def _download(
    *,
    repository: str,
    revision: str,
    destination: Path,
    allow_patterns: list[str],
    token: str,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is missing from the isolated IndicF5 environment."
        ) from error

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=destination,
        allow_patterns=allow_patterns,
        token=token,
    )


def _ensure_artifacts(
    *,
    repository: str,
    revision: str,
    destination: Path,
    expected: dict[str, str],
    token: str | None,
) -> None:
    failures = verify_artifacts(destination, expected)
    if not failures:
        return
    if not token:
        raise SystemExit(
            "HF_TOKEN is required because the qualified model artifacts are "
            f"not already present under {destination}."
        )
    _download(
        repository=repository,
        revision=revision,
        destination=destination,
        allow_patterns=list(expected),
        token=token,
    )
    failures = verify_artifacts(destination, expected)
    if failures:
        raise SystemExit("; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            os.environ.get(
                "VIDEO_TRANSLATOR_INDICF5_MODEL_ROOT",
                "/opt/video-translator/model-cache/indicf5-artifacts",
            )
        ),
    )
    parser.add_argument(
        "--vocoder-root",
        type=Path,
        default=Path(
            os.environ.get(
                "VIDEO_TRANSLATOR_INDICF5_VOCODER_ROOT",
                "/opt/video-translator/model-cache/vocos-mel-24khz",
            )
        ),
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )

    _ensure_artifacts(
        repository=INDICF5_REPOSITORY,
        revision=INDICF5_REVISION,
        destination=args.model_root,
        expected=INDICF5_ARTIFACTS,
        token=token,
    )
    _ensure_artifacts(
        repository=VOCODER_REPOSITORY,
        revision=VOCODER_REVISION,
        destination=args.vocoder_root,
        expected=VOCODER_ARTIFACTS,
        token=token,
    )
    print(
        "Verified qualified IndicF5 model and vocoder artifacts at "
        f"{args.model_root.parent}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

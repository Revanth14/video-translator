import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_gpu_bootstrap_script_documents_worker_runtime() -> None:
    script = ROOT / "scripts" / "bootstrap-gpu.sh"
    content = script.read_text(encoding="utf-8")

    assert script.is_file()
    assert "set -euo pipefail" in content
    assert "apt-get install" in content
    assert "ffmpeg" in content
    assert "uv sync" in content
    assert 'UV_BIN_DIR="${UV_BIN_DIR:-/usr/local/bin}"' in content
    assert "UV_INSTALL_DIR=\"$UV_BIN_DIR\"" in content
    assert "uv pip sync --python" in content
    assert "requirements-indicf5-gpu.txt" in content
    assert "cache-indicf5-models.py" in content
    assert "dub-mvp preflight" in content
    assert "dub-mvp worker --runs" in content


def test_gpu_runtime_lock_matches_the_qualified_environment() -> None:
    content = (ROOT / "requirements-indicf5-gpu.txt").read_text(
        encoding="utf-8"
    )

    assert "torch==2.13.0" in content
    assert "torchaudio==2.11.0" in content
    assert "torchcodec==0.13.0" in content
    assert "transformers==4.49.0" in content
    assert (
        "f5-tts @ git+https://github.com/AI4Bharat/IndicF5.git@"
        "13f7c4d627cc10111aea8fe9c0039462cacacdc7"
    ) in content


def test_model_cache_verifier_rejects_corrupt_artifacts(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "cache-indicf5-models.py"
    spec = importlib.util.spec_from_file_location("cache_indicf5_models", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"qualified-model")
    expected = {"model.bin": hashlib.sha256(b"qualified-model").hexdigest()}
    assert module.verify_artifacts(tmp_path, expected) == []

    artifact.write_bytes(b"corrupt")
    failures = module.verify_artifacts(tmp_path, expected)
    assert len(failures) == 1
    assert "checksum mismatch" in failures[0]


def test_env_example_contains_runner_and_provider_settings() -> None:
    content = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "VIDEO_TRANSLATOR_RUNNER=queued" in content
    assert "VIDEO_TRANSLATOR_RUNS=/srv/video-translator/runs" in content
    assert "OPENAI_API_KEY=" in content
    assert "WHISPERX_MODEL=large-v3" in content
    assert "INDICF5_MODEL=ai4bharat/IndicF5" in content
    assert "VIDEO_TRANSLATOR_INDICF5_PYTHON=" in content
    assert "VIDEO_TRANSLATOR_INDICF5_MODEL_ROOT=" in content
    assert "VIDEO_TRANSLATOR_INDICF5_VOCODER_ROOT=" in content
    assert (
        "VIDEO_TRANSLATOR_INDICF5_MODEL_REVISION="
        "ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
    ) in content


def test_gpu_droplet_runbook_has_operator_smoke_test() -> None:
    content = (ROOT / "docs" / "gpu-droplet-runbook.md").read_text(
        encoding="utf-8"
    )

    assert "uv run dub-mvp web" in content
    assert "--runner queued" in content
    assert "uv run dub-mvp worker" in content
    assert "--once" in content
    assert "Operator Smoke Test" in content

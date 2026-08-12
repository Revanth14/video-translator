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
    assert "dub-mvp preflight" in content
    assert "dub-mvp worker --runs" in content


def test_env_example_contains_runner_and_provider_settings() -> None:
    content = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "VIDEO_TRANSLATOR_RUNNER=queued" in content
    assert "VIDEO_TRANSLATOR_RUNS=/srv/video-translator/runs" in content
    assert "OPENAI_API_KEY=" in content
    assert "WHISPERX_MODEL=large-v3" in content
    assert "INDICF5_MODEL=ai4bharat/IndicF5" in content


def test_gpu_droplet_runbook_has_operator_smoke_test() -> None:
    content = (ROOT / "docs" / "gpu-droplet-runbook.md").read_text(
        encoding="utf-8"
    )

    assert "uv run dub-mvp web" in content
    assert "--runner queued" in content
    assert "uv run dub-mvp worker" in content
    assert "--once" in content
    assert "Operator Smoke Test" in content

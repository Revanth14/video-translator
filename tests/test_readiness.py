from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from dub_mvp.artifacts import (
    completed_artifact_metadata,
    write_artifact_metadata,
)
from dub_mvp.benchmark import BenchmarkReport, EvidenceStatus
from dub_mvp.cli import app
from dub_mvp.configuration import (
    build_configuration_snapshot,
    write_configuration_snapshot,
)
from dub_mvp.manifest import RunManifest
from dub_mvp.readiness import (
    DeploymentTarget,
    ReadinessStatus,
    assess_deployment_readiness,
    assess_language_expansion,
    assess_research_readiness,
)


def _run_with_benchmark(tmp_path: Path, *, passed: bool = True) -> Path:
    run = tmp_path / "run"
    manifest = RunManifest(
        run_id="quality-baseline",
        source_path=str(run / "input.mp4"),
        source_start_ms=0,
        source_end_ms=2_000_000,
        source_language="en",
        target_language="hi",
    )
    manifest.outputs.update(
        write_configuration_snapshot(
            build_configuration_snapshot(
                run_directory=run,
                source_language="en",
                target_language="hi",
            ),
            run_directory=run,
        )
    )
    manifest.save(run)
    report = BenchmarkReport(
        run_id=manifest.run_id,
        benchmark_fingerprint="f" * 64,
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        snapshot_manifest_revision=manifest.revision,
        benchmark_scope={},
        input_media={},
        transcription={},
        translation={},
        synthesis={},
        timing={},
        rendering={},
        stages={},
        resources={},
        cost={},
        storage={},
        integrity={},
        human_review={},
        quality_gates=[],
        release_gate_status=(
            EvidenceStatus.PASSED if passed else EvidenceStatus.NOT_MEASURED
        ),
        missing_evidence=[] if passed else ["human review"],
    )
    benchmark = run / "benchmark" / "benchmark.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    metadata = benchmark.with_name(benchmark.name + ".meta.json")
    write_artifact_metadata(
        metadata,
        completed_artifact_metadata(
            artifact_id="benchmark",
            kind="benchmark_json",
            path=benchmark,
            root=run,
            inputs={"baseline": manifest.run_id},
        ),
    )
    manifest = RunManifest.load(run)
    manifest.outputs.update(
        {
            "benchmark_json": str(benchmark),
            "benchmark_json_metadata": str(metadata),
        }
    )
    manifest.save(run)
    return run


def test_local_release_requires_and_accepts_verified_passing_benchmark(
    tmp_path: Path,
) -> None:
    run = _run_with_benchmark(tmp_path)

    report = assess_deployment_readiness(
        run,
        target=DeploymentTarget.LOCAL,
    )

    assert report.status == ReadinessStatus.PASSED
    assert all(check.status == ReadinessStatus.PASSED for check in report.checks)


def test_corrupt_benchmark_blocks_release(tmp_path: Path) -> None:
    run = _run_with_benchmark(tmp_path)
    manifest = RunManifest.load(run)
    Path(manifest.outputs["benchmark_json"]).write_text("{}", encoding="utf-8")

    report = assess_deployment_readiness(
        run,
        target=DeploymentTarget.LOCAL,
    )

    assert report.status == ReadinessStatus.BLOCKED
    integrity = next(
        item for item in report.checks if item.check_id == "benchmark_integrity"
    )
    assert integrity.status == ReadinessStatus.BLOCKED
    assert "checksum" in integrity.detail or "size" in integrity.detail


def test_aws_remains_blocked_until_remote_durability_and_cost_are_measured(
    tmp_path: Path,
) -> None:
    run = _run_with_benchmark(tmp_path)

    report = assess_deployment_readiness(
        run,
        target=DeploymentTarget.AWS,
    )

    assert report.status == ReadinessStatus.BLOCKED
    statuses = {item.check_id: item.status for item in report.checks}
    assert statuses["worker_container"] == ReadinessStatus.PASSED
    assert statuses["configuration_snapshot"] == ReadinessStatus.PASSED
    assert statuses["gpu_runtime_dependencies"] == ReadinessStatus.BLOCKED
    assert statuses["remote_artifact_store"] == ReadinessStatus.BLOCKED
    assert statuses["remote_conditional_state"] == ReadinessStatus.BLOCKED
    assert statuses["measured_cloud_cost"] == ReadinessStatus.BLOCKED


def test_language_expansion_requires_hindi_gate_and_matching_evaluation_set(
    tmp_path: Path,
) -> None:
    run = _run_with_benchmark(tmp_path)
    evaluation = tmp_path / "ta-evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_language": "en",
                "target_language": "ta",
                "version": "2026-08-13",
                "items": [
                    {
                        "item_id": "sample-1",
                        "source_text": "Hello",
                        "reference_text": "Vanakkam",
                        "coverage_tags": ["greeting"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ready = assess_language_expansion(
        run,
        candidate_language="ta",
        evaluation_set_path=evaluation,
    )
    missing = assess_language_expansion(
        run,
        candidate_language="ta",
        evaluation_set_path=None,
    )

    assert ready.status == ReadinessStatus.PASSED
    assert missing.status == ReadinessStatus.BLOCKED


def test_research_gate_requires_x_y_z_w_e_and_existing_evaluation(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_metric": "median semantic adequacy",
                "baseline_value": 3.8,
                "measured_bottleneck": "named-entity pronunciation",
                "bottleneck_evidence": ["14/50 reviewed names failed"],
                "training_objective": "reduce name pronunciation defects",
                "expected_improvement": "critical defects below 2%",
                "evaluation_method": "blind review on held-out names",
                "evaluation_set": "evaluation.json",
            }
        ),
        encoding="utf-8",
    )

    report = assess_research_readiness(decision)

    assert report.status == ReadinessStatus.PASSED


def test_release_check_cli_exits_nonzero_when_evidence_is_missing(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    RunManifest(
        run_id="missing-evidence",
        source_path="input.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    ).save(run)

    result = CliRunner().invoke(
        app,
        ["release-check", "--run", str(run), "--target", "local"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"

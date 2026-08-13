from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dub_mvp.artifacts import ArtifactMetadata, verify_artifact_integrity
from dub_mvp.benchmark import BenchmarkReport, EvidenceStatus
from dub_mvp.configuration import PipelineConfigurationSnapshot
from dub_mvp.manifest import RunManifest


class ReadinessError(RuntimeError):
    retryable = False


class ReadinessStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class DeploymentTarget(str, Enum):
    LOCAL = "local"
    AWS = "aws"


class ReadinessCheck(BaseModel):
    check_id: str
    status: ReadinessStatus
    detail: str


class DeploymentReadiness(BaseModel):
    schema_version: int = 1
    run_id: str
    target: DeploymentTarget
    status: ReadinessStatus
    checks: list[ReadinessCheck]


class LanguageEvaluationItem(BaseModel):
    item_id: str
    source_text: str
    reference_text: str
    coverage_tags: list[str] = Field(min_length=1)

    @field_validator("item_id", "source_text", "reference_text")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Evaluation-set text fields cannot be empty.")
        return cleaned


class LanguageEvaluationSet(BaseModel):
    schema_version: int = 1
    source_language: str
    target_language: str
    version: str
    items: list[LanguageEvaluationItem] = Field(min_length=1)

    @model_validator(mode="after")
    def stable_unique_ids(self) -> "LanguageEvaluationSet":
        identifiers = [item.item_id for item in self.items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evaluation-set item IDs must be unique.")
        return self


class LanguageExpansionReadiness(BaseModel):
    schema_version: int = 1
    baseline_run_id: str
    candidate_language: str
    status: ReadinessStatus
    checks: list[ReadinessCheck]


class ModelResearchDecision(BaseModel):
    schema_version: int = 1
    baseline_metric: str
    baseline_value: float
    measured_bottleneck: str
    bottleneck_evidence: list[str] = Field(min_length=1)
    training_objective: str
    expected_improvement: str
    evaluation_method: str
    evaluation_set: str

    @field_validator(
        "baseline_metric",
        "measured_bottleneck",
        "training_objective",
        "expected_improvement",
        "evaluation_method",
        "evaluation_set",
    )
    @classmethod
    def decision_text_is_required(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Research decision fields cannot be empty.")
        return cleaned


class ResearchReadiness(BaseModel):
    schema_version: int = 1
    status: ReadinessStatus
    checks: list[ReadinessCheck]


def assess_deployment_readiness(
    run_directory: Path,
    *,
    target: DeploymentTarget,
    project_root: Path | None = None,
) -> DeploymentReadiness:
    run_directory = run_directory.resolve()
    manifest = RunManifest.load(run_directory)
    benchmark, benchmark_checks = _verified_benchmark(manifest, run_directory)
    checks = list(benchmark_checks)
    checks.append(
        _check(
            "benchmark_release_gate",
            benchmark is not None
            and benchmark.release_gate_status == EvidenceStatus.PASSED,
            (
                "The benchmark first-release gate passed."
                if benchmark is not None
                and benchmark.release_gate_status == EvidenceStatus.PASSED
                else "A verified benchmark with release_gate_status=passed is required."
            ),
        )
    )
    if target == DeploymentTarget.AWS:
        root = project_root or Path(__file__).resolve().parents[2]
        checks.extend(_verified_configuration(manifest, run_directory))
        checks.append(
            _check(
                "worker_container",
                (root / "Dockerfile").is_file()
                and (root / "docker" / "worker-entrypoint.sh").is_file(),
                "The existing worker executor has a container definition.",
            )
        )
        # These checks intentionally block deployment. AGENTS.md and the build
        # plan prohibit adding cloud services before a passing measured
        # benchmark; none of these capabilities may be inferred from a local
        # filesystem implementation.
        checks.extend(
            [
                _check(
                    "gpu_runtime_dependencies",
                    False,
                    "The container does not yet install benchmark-selected "
                    "GPU/provider dependencies.",
                ),
                _check(
                    "remote_artifact_store",
                    False,
                    "No measured need has authorized an S3 artifact backend.",
                ),
                _check(
                    "remote_conditional_state",
                    False,
                    "No remote state store with conditional writes/fencing exists.",
                ),
                _check(
                    "incremental_remote_upload",
                    False,
                    "Completed artifacts are not incrementally uploaded remotely.",
                ),
                _check(
                    "measured_cloud_cost",
                    False,
                    "GPU shape and cloud cost have not been measured.",
                ),
            ]
        )
    status = (
        ReadinessStatus.PASSED
        if all(item.status == ReadinessStatus.PASSED for item in checks)
        else ReadinessStatus.BLOCKED
    )
    return DeploymentReadiness(
        run_id=manifest.run_id,
        target=target,
        status=status,
        checks=checks,
    )


def assess_language_expansion(
    baseline_run_directory: Path,
    *,
    candidate_language: str,
    evaluation_set_path: Path | None,
) -> LanguageExpansionReadiness:
    baseline_run_directory = baseline_run_directory.resolve()
    manifest = RunManifest.load(baseline_run_directory)
    candidate = candidate_language.strip().lower()
    language_valid = bool(re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", candidate))
    checks = [
        _check(
            "candidate_language",
            language_valid and candidate != manifest.target_language.lower(),
            (
                f"Candidate language {candidate!r} is distinct and well formed."
                if language_valid and candidate != manifest.target_language.lower()
                else "Candidate must be a distinct BCP-47-style language code."
            ),
        )
    ]
    benchmark, benchmark_checks = _verified_benchmark(
        manifest, baseline_run_directory
    )
    checks.extend(benchmark_checks)
    checks.append(
        _check(
            "hindi_quality_gate",
            manifest.target_language.lower() == "hi"
            and benchmark is not None
            and benchmark.release_gate_status == EvidenceStatus.PASSED,
            "Hindi must have a verified passing first-release benchmark.",
        )
    )
    evaluation: LanguageEvaluationSet | None = None
    evaluation_error: str | None = None
    if evaluation_set_path is not None:
        try:
            evaluation = LanguageEvaluationSet.model_validate_json(
                evaluation_set_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            evaluation_error = str(error)
    checks.append(
        _check(
            "candidate_evaluation_set",
            evaluation is not None
            and evaluation.target_language.lower() == candidate
            and evaluation.source_language.lower() == manifest.source_language.lower(),
            (
                "Candidate evaluation set is valid and language-aligned."
                if evaluation is not None
                and evaluation.target_language.lower() == candidate
                and evaluation.source_language.lower()
                == manifest.source_language.lower()
                else "A valid source/candidate-aligned evaluation set is required"
                + (f": {evaluation_error}" if evaluation_error else ".")
            ),
        )
    )
    status = (
        ReadinessStatus.PASSED
        if all(item.status == ReadinessStatus.PASSED for item in checks)
        else ReadinessStatus.BLOCKED
    )
    return LanguageExpansionReadiness(
        baseline_run_id=manifest.run_id,
        candidate_language=candidate,
        status=status,
        checks=checks,
    )


def assess_research_readiness(decision_path: Path) -> ResearchReadiness:
    checks: list[ReadinessCheck] = []
    try:
        decision = ModelResearchDecision.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        return ResearchReadiness(
            status=ReadinessStatus.BLOCKED,
            checks=[
                _check(
                    "research_decision",
                    False,
                    f"Decision does not state X/Y/Z/W/E: {error}",
                )
            ],
        )
    evaluation_path = Path(decision.evaluation_set)
    if not evaluation_path.is_absolute():
        evaluation_path = decision_path.parent / evaluation_path
    checks.extend(
        [
            _check(
                "research_decision",
                True,
                "Baseline, bottleneck, objective, expected improvement, and "
                "evaluation method are explicit.",
            ),
            _check(
                "research_evaluation_set",
                evaluation_path.is_file(),
                (
                    f"Evaluation set exists: {evaluation_path}"
                    if evaluation_path.is_file()
                    else f"Evaluation set is missing: {evaluation_path}"
                ),
            ),
        ]
    )
    status = (
        ReadinessStatus.PASSED
        if all(item.status == ReadinessStatus.PASSED for item in checks)
        else ReadinessStatus.BLOCKED
    )
    return ResearchReadiness(status=status, checks=checks)


def readiness_json(report: BaseModel) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2) + "\n"


def _verified_benchmark(
    manifest: RunManifest,
    run_directory: Path,
) -> tuple[BenchmarkReport | None, list[ReadinessCheck]]:
    report_path = manifest.outputs.get("benchmark_json")
    metadata_path = manifest.outputs.get("benchmark_json_metadata")
    declared = bool(report_path and metadata_path)
    checks = [
        _check(
            "benchmark_declared",
            declared,
            "Manifest declares benchmark JSON and its proof sidecar.",
        )
    ]
    if not declared:
        return None, checks
    assert report_path is not None and metadata_path is not None
    try:
        report_file = _run_path(run_directory, report_path)
        metadata_file = _run_path(run_directory, metadata_path)
        metadata = ArtifactMetadata.model_validate_json(
            metadata_file.read_text(encoding="utf-8")
        )
        verification = verify_artifact_integrity(metadata, root=run_directory)
        if (run_directory / metadata.path).resolve() != report_file.resolve():
            verification.valid = False
            verification.reason = "sidecar points to a different artifact"
        if not verification.valid:
            checks.append(
                _check(
                    "benchmark_integrity",
                    False,
                    f"Benchmark proof failed: {verification.reason}",
                )
            )
            return None, checks
        report = BenchmarkReport.model_validate_json(
            report_file.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        checks.append(
            _check("benchmark_integrity", False, f"Invalid benchmark proof: {error}")
        )
        return None, checks
    checks.append(
        _check(
            "benchmark_integrity",
            True,
            "Benchmark checksum, size, and completed status are valid.",
        )
    )
    return report, checks


def _verified_configuration(
    manifest: RunManifest,
    run_directory: Path,
) -> list[ReadinessCheck]:
    snapshot_path = manifest.outputs.get("configuration_snapshot")
    metadata_path = manifest.outputs.get("configuration_snapshot_metadata")
    if not snapshot_path or not metadata_path:
        return [
            _check(
                "configuration_snapshot",
                False,
                "Manifest does not declare a configuration snapshot and proof.",
            )
        ]
    try:
        snapshot_file = _run_path(run_directory, snapshot_path)
        metadata_file = _run_path(run_directory, metadata_path)
        snapshot = PipelineConfigurationSnapshot.model_validate_json(
            snapshot_file.read_text(encoding="utf-8")
        )
        metadata = ArtifactMetadata.model_validate_json(
            metadata_file.read_text(encoding="utf-8")
        )
        verification = verify_artifact_integrity(metadata, root=run_directory)
        aligned = (
            snapshot.source_language == manifest.source_language
            and snapshot.target_language == manifest.target_language
            and (run_directory / metadata.path).resolve()
            == snapshot_file.resolve()
        )
    except (OSError, ValidationError, ValueError) as error:
        return [
            _check(
                "configuration_snapshot",
                False,
                f"Configuration snapshot is invalid: {error}",
            )
        ]
    return [
        _check(
            "configuration_snapshot",
            verification.valid and aligned,
            (
                "Configuration snapshot is checksum-verified and language-aligned."
                if verification.valid and aligned
                else "Configuration snapshot proof or manifest alignment failed."
            ),
        )
    ]


def _run_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _check(check_id: str, passed: bool, detail: str) -> ReadinessCheck:
    return ReadinessCheck(
        check_id=check_id,
        status=ReadinessStatus.PASSED if passed else ReadinessStatus.BLOCKED,
        detail=detail,
    )

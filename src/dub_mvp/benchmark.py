from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dub_mvp.artifacts import (
    ArtifactMetadata,
    completed_artifact_metadata,
    fingerprint_inputs,
    relative_artifact_path,
    sha256_file,
    verify_artifact,
    write_artifact_metadata,
)
from dub_mvp.manifest import (
    PIPELINE_STAGE_NAMES,
    MutationAborted,
    RunManifest,
    StageStatus,
    mutate_manifest,
)
from dub_mvp.observability import RunStatusDocument, build_run_status
from dub_mvp.render import RenderReport


class BenchmarkError(RuntimeError):
    retryable = False


class EvidenceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_MEASURED = "not_measured"
    NOT_APPLICABLE = "not_applicable"


class QualityGate(BaseModel):
    gate_id: str
    description: str
    status: EvidenceStatus
    observed: Any = None
    required: Any = None
    evidence: list[str] = Field(default_factory=list)


class HumanReviewScores(BaseModel):
    semantic_adequacy: int = Field(ge=1, le=5)
    naturalness: int = Field(ge=1, le=5)
    pronunciation: int = Field(ge=1, le=5)
    timing_quality: int = Field(ge=1, le=5)
    speaker_consistency: int = Field(ge=1, le=5)
    overall_usability: int = Field(ge=1, le=5)


class HumanReviewDefect(BaseModel):
    category: Literal[
        "mistranslation",
        "semantic_omission",
        "incorrect_name",
        "pronunciation",
        "timing",
        "speaker_consistency",
        "missing_audio",
        "duplicate_audio",
        "other",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    utterance_id: str | None = None

    @field_validator("description")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Human-review defect fields cannot be empty.")
        return cleaned


class HumanReviewSample(BaseModel):
    utterance_id: str
    coverage_tags: list[str]
    scores: HumanReviewScores
    notes: str | None = None

    @model_validator(mode="after")
    def validate_tags(self) -> "HumanReviewSample":
        if not self.coverage_tags:
            raise ValueError("Human-review sample requires coverage tags.")
        if len(self.coverage_tags) != len(set(self.coverage_tags)):
            raise ValueError("Human-review coverage tags must be unique.")
        return self


class SourceConditions(BaseModel):
    noise_present: bool
    music_present: bool
    overlap_present: bool


class HumanReviewSubmission(BaseModel):
    schema_version: int = 1
    reviewer: str
    reviewed_at: datetime
    source_conditions: SourceConditions
    samples: list[HumanReviewSample]
    critical_defects: list[HumanReviewDefect] = Field(default_factory=list)

    @field_validator("reviewer")
    @classmethod
    def reviewer_is_required(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Human-review reviewer is required.")
        return cleaned

    @model_validator(mode="after")
    def validate_samples(self) -> "HumanReviewSubmission":
        identifiers = [item.utterance_id for item in self.samples]
        if not identifiers:
            raise ValueError("Human review must contain at least one sample.")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Human review contains duplicate utterance IDs.")
        return self


class BenchmarkReport(BaseModel):
    schema_version: int = 1
    run_id: str
    benchmark_fingerprint: str
    generated_at: datetime
    snapshot_manifest_revision: int
    benchmark_scope: dict[str, Any]
    input_media: dict[str, Any]
    transcription: dict[str, Any]
    translation: dict[str, Any]
    synthesis: dict[str, Any]
    timing: dict[str, Any]
    rendering: dict[str, Any]
    stages: dict[str, Any]
    resources: dict[str, Any]
    cost: dict[str, Any]
    storage: dict[str, Any]
    integrity: dict[str, Any]
    human_review: dict[str, Any]
    quality_gates: list[QualityGate]
    release_gate_status: EvidenceStatus
    missing_evidence: list[str]


class BenchmarkArtifacts(BaseModel):
    benchmark_json: str
    benchmark_json_metadata: str
    benchmark_markdown: str
    benchmark_markdown_metadata: str
    human_review_template: str
    human_review_template_metadata: str

    def as_outputs(self, root: Path) -> dict[str, str]:
        return {
            name: str((root / value).resolve())
            for name, value in self.model_dump().items()
        }


def build_benchmark(
    run_directory: Path,
    *,
    human_review_path: Path | None = None,
    reuse_completed: bool = True,
) -> tuple[BenchmarkReport, BenchmarkArtifacts]:
    run_directory = run_directory.resolve()
    manifest = RunManifest.load(run_directory)
    status = build_run_status(run_directory)
    review = _load_human_review(human_review_path)
    inputs = _benchmark_inputs(
        manifest=manifest,
        run_directory=run_directory,
        human_review_path=human_review_path,
    )
    fingerprint = fingerprint_inputs(inputs)
    benchmark_directory = run_directory / "benchmark"
    benchmark_directory.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark-{fingerprint[:16]}"

    if reuse_completed:
        reusable = _find_reusable_benchmark(
            directory=benchmark_directory,
            stem=stem,
            expected_inputs=inputs,
            root=run_directory,
        )
        if reusable is not None:
            report, artifacts = reusable
            _publish_benchmark_outputs(run_directory, artifacts)
            return report, artifacts

    revision = _next_revision(benchmark_directory, stem)
    label = f"r{revision:04d}"
    json_path = benchmark_directory / f"{stem}-{label}.json"
    markdown_path = benchmark_directory / f"{stem}-{label}.md"
    template_path = benchmark_directory / f"human-review-{fingerprint[:16]}-{label}.json"
    template = _human_review_template(manifest, run_directory)
    _write_json(template_path, template)

    report = _aggregate_report(
        manifest=manifest,
        status=status,
        run_directory=run_directory,
        fingerprint=fingerprint,
        review=review,
        template=template,
    )
    _write_json(json_path, report.model_dump(mode="json"))
    _write_text(markdown_path, benchmark_markdown(report))

    json_metadata = _metadata_path(json_path)
    markdown_metadata = _metadata_path(markdown_path)
    template_metadata = _metadata_path(template_path)
    for path, metadata_path, kind in (
        (json_path, json_metadata, "benchmark_json"),
        (markdown_path, markdown_metadata, "benchmark_markdown"),
        (template_path, template_metadata, "human_review_template"),
    ):
        write_artifact_metadata(
            metadata_path,
            completed_artifact_metadata(
                artifact_id=f"{kind}_{label}",
                kind=kind,
                path=path,
                root=run_directory,
                inputs=_artifact_inputs(inputs, kind),
                provider="internal",
                configuration={"benchmark_fingerprint": fingerprint},
            ),
        )
    artifacts = BenchmarkArtifacts(
        benchmark_json=relative_artifact_path(json_path, run_directory),
        benchmark_json_metadata=relative_artifact_path(
            json_metadata, run_directory
        ),
        benchmark_markdown=relative_artifact_path(markdown_path, run_directory),
        benchmark_markdown_metadata=relative_artifact_path(
            markdown_metadata, run_directory
        ),
        human_review_template=relative_artifact_path(template_path, run_directory),
        human_review_template_metadata=relative_artifact_path(
            template_metadata, run_directory
        ),
    )
    _publish_benchmark_outputs(run_directory, artifacts)
    return report, artifacts


def benchmark_markdown(report: BenchmarkReport) -> str:
    lines = [
        f"# Benchmark — {report.run_id}",
        "",
        f"Release gate: **{report.release_gate_status.value}**",
        "",
        "## Scope",
        "",
        f"- Duration: {report.benchmark_scope['duration_minutes']:.2f} minutes",
        f"- Long-form 30–45 minute qualification: {report.benchmark_scope['long_form_qualified']}",
        f"- Manifest revision sampled: {report.snapshot_manifest_revision}",
        "",
        "## Core measurements",
        "",
        f"- Utterances: {report.integrity['source_utterance_count']}",
        f"- Missing localized/synthesized: {len(report.integrity['missing_utterance_ids'])}",
        f"- Duplicate IDs: {len(report.integrity['duplicate_utterance_ids'])}",
        f"- Primary timing tolerance: {_display(report.timing.get('within_primary_percent'), suffix='%')}",
        f"- Hard timing tolerance: {_display(report.timing.get('within_hard_percent'), suffix='%')}",
        f"- Median absolute timing error: {_display(report.timing.get('median_absolute_error_ms'), suffix=' ms')}",
        f"- P95 absolute timing error: {_display(report.timing.get('p95_absolute_error_ms'), suffix=' ms')}",
        f"- Total wall time: {_display(report.stages.get('total_wall_seconds'), suffix=' s')}",
        f"- Reported external cost: {_display(report.cost.get('reported_external_usd'), prefix='$')}",
        f"- Storage: {report.storage['bytes']} bytes",
        "",
        "## Quality gates",
        "",
        "| Gate | Status | Observed | Required |",
        "|---|---|---:|---:|",
    ]
    for gate in report.quality_gates:
        lines.append(
            f"| {gate.description} | {gate.status.value} | "
            f"{_markdown_cell(gate.observed)} | {_markdown_cell(gate.required)} |"
        )
    lines.extend(["", "## Missing evidence", ""])
    if report.missing_evidence:
        lines.extend(f"- {item}" for item in report.missing_evidence)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Human review",
            "",
            f"- Status: {report.human_review['status']}",
            f"- Samples: {report.human_review['sample_count']}",
            f"- Median semantic adequacy: {_display(report.human_review.get('median_semantic_adequacy'))}",
            f"- Critical defects: {report.human_review['critical_defect_count']}",
            "",
            "This report never treats unavailable GPU, provider, or human-review data as zero.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_report(
    *,
    manifest: RunManifest,
    status: RunStatusDocument,
    run_directory: Path,
    fingerprint: str,
    review: HumanReviewSubmission | None,
    template: dict[str, Any],
) -> BenchmarkReport:
    transcript = _read_json(manifest.outputs.get("transcript"))
    translation_segments = _read_list(
        manifest.outputs.get("translation_segments")
        or manifest.outputs.get("segments")
    )
    localized = _read_list(manifest.outputs.get("localized_segments"))
    synthesized = _read_list(manifest.outputs.get("synthesized_segments"))
    translation_metrics = _read_json(manifest.outputs.get("translation_metrics"))
    synthesis_metrics = _read_json(manifest.outputs.get("synthesis_metrics"))
    duration_metrics = _read_json(manifest.outputs.get("duration_metrics"))
    corrections = _read_list(manifest.outputs.get("duration_corrections"))
    render_report = _load_render_report(manifest.outputs.get("render_report"))
    render_report_declared = bool(manifest.outputs.get("render_report"))
    input_seconds = manifest.duration_ms / 1000
    duration_minutes = input_seconds / 60

    source_ids = _ids(translation_segments)
    localized_ids = _ids(localized)
    synthesized_ids = _ids(synthesized)
    duration_ids = [str(item.get("utterance_id")) for item in corrections if item.get("utterance_id")]
    duplicate_ids = sorted(
        set(_duplicates(source_ids))
        | set(_duplicates(localized_ids))
        | set(_duplicates(synthesized_ids))
        | set(_duplicates(duration_ids))
    )
    source_set = set(source_ids)
    missing_ids = sorted(
        (source_set - set(localized_ids))
        | (source_set - set(synthesized_ids))
        | (source_set - set(duration_ids) if corrections else set())
    )
    unknown_ids = sorted(
        (set(localized_ids) | set(synthesized_ids) | set(duration_ids)) - source_set
    )
    voice_consistency = _voice_consistency(synthesized)
    errors_ms = [
        abs(int(item["duration_error_ms"]))
        for item in corrections
        if item.get("duration_error_ms") is not None
    ]
    ratios = [
        abs(float(item["duration_ratio"]) - 1)
        for item in corrections
        if item.get("duration_ratio") is not None
    ]
    confidences = _transcript_confidences(transcript)
    human = _human_review_summary(review, template)
    storage_bytes = _run_storage_bytes(run_directory)
    source_path = _resolve_source(manifest.source_path, run_directory)
    source_size = source_path.stat().st_size if source_path and source_path.is_file() else None
    stage_rows = {
        name: {
            "status": manifest.stages[name].status.value,
            "attempt_count": manifest.stages[name].attempt_count,
            "duration_seconds": manifest.stages[name].duration_seconds,
            "provider": manifest.stages[name].provider,
            "model": manifest.stages[name].model,
            "resources": (
                manifest.stages[name].resources.model_dump(mode="json")
                if manifest.stages[name].resources
                else None
            ),
            "cost_usd": manifest.stages[name].cost_usd,
        }
        for name in PIPELINE_STAGE_NAMES
    }
    total_wall = sum(
        item.duration_seconds or 0 for item in manifest.stages.values()
    )
    reported_costs = [
        item.cost_usd
        for item in manifest.stages.values()
        if item.cost_usd is not None
    ]
    render_validation = (
        render_report.validation.model_dump(mode="json")
        if render_report
        else None
    )
    interrupted = _has_interruption_evidence(status)
    artifact_reuse = _artifact_reuse_evidence(
        translation_metrics, synthesis_metrics
    )

    gates = [
        _gate(
            "long_form",
            "30–45 minute benchmark input",
            30 <= duration_minutes <= 45,
            round(duration_minutes, 3),
            "30–45 minutes",
        ),
        _gate(
            "utterance_integrity",
            "Zero missing, duplicated, or unknown utterances",
            not missing_ids and not duplicate_ids and not unknown_ids and bool(source_ids),
            {
                "missing": len(missing_ids),
                "duplicate": len(duplicate_ids),
                "unknown": len(unknown_ids),
            },
            0,
        ),
        _gate(
            "silent_failures",
            "Zero silent stage failures",
            all(manifest.stages[name].status == StageStatus.COMPLETED for name in PIPELINE_STAGE_NAMES)
            and not any(
                attempt.status == StageStatus.RUNNING
                for record in manifest.stages.values()
                for attempt in record.attempts
            )
            and not any(
                item.status == "running"
                for group in status.work_items.values()
                for item in group
            ),
            {
                "incomplete_stages": [
                    name for name in PIPELINE_STAGE_NAMES
                    if manifest.stages[name].status != StageStatus.COMPLETED
                ],
                "structured_errors": len(manifest.error_records),
                "running_attempts": sum(
                    attempt.status == StageStatus.RUNNING
                    for record in manifest.stages.values()
                    for attempt in record.attempts
                ),
                "running_work_items": sum(
                    item.status == "running"
                    for group in status.work_items.values()
                    for item in group
                ),
            },
            0,
        ),
        QualityGate(
            gate_id="interruption_recovery",
            description="Successful recovery after worker interruption",
            status=(EvidenceStatus.PASSED if interrupted else EvidenceStatus.NOT_MEASURED),
            observed=interrupted,
            required=True,
            evidence=["attempt history with error_class=interrupted"] if interrupted else [],
        ),
        QualityGate(
            gate_id="artifact_reuse",
            description="Verified completed artifacts were reused",
            status=(EvidenceStatus.PASSED if artifact_reuse else EvidenceStatus.NOT_MEASURED),
            observed=artifact_reuse,
            required=True,
        ),
        _gate(
            "voice_consistency",
            "100% consistent speaker-to-voice mapping",
            bool(synthesized) and voice_consistency["inconsistent_speaker_count"] == 0,
            voice_consistency,
            "0 inconsistent speakers",
        ),
        _optional_numeric_gate(
            "primary_timing",
            "At least 90% meet primary timing tolerance",
            duration_metrics.get("within_primary_percent"),
            minimum=90,
            required="≥90%",
        ),
        _gate(
            "cumulative_drift",
            "No severe cumulative A/V drift",
            # Start alignment is structural (utterances are muxed at immutable
            # source offsets), so this gate tests what is actually measurable:
            # output duration error, and audio that never runs into the next
            # utterance's cue.
            bool(render_validation)
            and render_validation.get("duration_within_tolerance") is True
            and render_validation.get("unintended_overlap_count") == 0
            and not render_validation.get("missing_utterance_ids")
            and not render_validation.get("duplicate_utterance_ids"),
            (
                {
                    "duration_error_ms": render_validation.get("duration_error_ms"),
                    "unintended_overlap_count": render_validation.get(
                        "unintended_overlap_count"
                    ),
                    "start_alignment_basis": render_validation.get(
                        "start_alignment_basis"
                    ),
                }
                if render_validation
                else None
            ),
            "output within tolerance, no overlap, no missing or duplicate audio",
            measured=render_validation is not None,
        ),
        _optional_numeric_gate(
            "semantic_adequacy",
            "Median semantic adequacy is at least 4/5",
            human.get("median_semantic_adequacy"),
            minimum=4,
            required="≥4/5",
        ),
        QualityGate(
            gate_id="critical_mistranslations",
            description="No critical mistranslations",
            status=(
                EvidenceStatus.NOT_MEASURED
                if review is None
                else (
                    EvidenceStatus.PASSED
                    if human["critical_mistranslation_count"] == 0
                    else EvidenceStatus.FAILED
                )
            ),
            observed=(
                None if review is None else human["critical_mistranslation_count"]
            ),
            required=0,
        ),
        QualityGate(
            gate_id="complete_reporting",
            description="Complete runtime, GPU/resource, retry, and cost reporting",
            status=EvidenceStatus.NOT_MEASURED,
            observed={
                "stage_runtime": all(item.duration_seconds is not None for item in manifest.stages.values()),
                "process_resources": all(item.resources is not None for item in manifest.stages.values()),
                "gpu_time": None,
                "peak_vram": None,
                "gpu_cost": None,
                "external_cost": sum(reported_costs) if reported_costs else None,
            },
            required="all fields measured",
            evidence=["GPU time/VRAM/cost are not instrumented yet"],
        ),
        _gate(
            "usable_output",
            "Output is validated and usable without reconstruction",
            bool(render_report and render_report.validation.passed),
            render_validation,
            True,
            measured=render_report_declared,
        ),
        QualityGate(
            gate_id="human_sample_coverage",
            description="Human review covers every required sample category",
            status=(
                EvidenceStatus.NOT_MEASURED
                if review is None
                else (
                    EvidenceStatus.PASSED
                    if not human["missing_coverage_tags"]
                    else EvidenceStatus.FAILED
                )
            ),
            observed=human.get("covered_tags"),
            required=human.get("required_tags"),
            evidence=(
                ["Missing: " + ", ".join(human["missing_coverage_tags"])]
                if human["missing_coverage_tags"]
                else []
            ),
        ),
    ]
    missing_evidence = [
        gate.description
        for gate in gates
        if gate.status == EvidenceStatus.NOT_MEASURED
    ]
    release_status = (
        EvidenceStatus.FAILED
        if any(gate.status == EvidenceStatus.FAILED for gate in gates)
        else (
            EvidenceStatus.NOT_MEASURED
            if missing_evidence
            else EvidenceStatus.PASSED
        )
    )
    return BenchmarkReport(
        run_id=manifest.run_id,
        benchmark_fingerprint=fingerprint,
        generated_at=datetime.now(timezone.utc),
        snapshot_manifest_revision=manifest.revision,
        benchmark_scope={
            "duration_minutes": round(duration_minutes, 4),
            "long_form_qualified": 30 <= duration_minutes <= 45,
            "target_range_minutes": [30, 45],
        },
        input_media={
            "duration_ms": manifest.duration_ms,
            "size_bytes": source_size,
            "format_name": manifest.media.format_name if manifest.media else None,
            "video_codec": manifest.media.video_codec if manifest.media else None,
            "width": manifest.media.width if manifest.media else None,
            "height": manifest.media.height if manifest.media else None,
            "frame_rate": manifest.media.frame_rate if manifest.media else None,
            "audio_codec": manifest.media.audio_codec if manifest.media else None,
            "audio_channels": manifest.media.audio_channels if manifest.media else None,
            "audio_sample_rate": manifest.media.audio_sample_rate if manifest.media else None,
        },
        transcription={
            "wall_seconds": manifest.stages["transcribe"].duration_seconds,
            "real_time_factor": (
                manifest.stages["transcribe"].duration_seconds / input_seconds
                if manifest.stages["transcribe"].duration_seconds is not None
                else None
            ),
            "word_confidence_mean": (
                round(statistics.fmean(confidences), 6) if confidences else None
            ),
            "word_confidence_p10": _percentile(confidences, 10),
            "speaker_count": len(
                {
                    utterance.get("speaker_id")
                    for utterance in transcript.get("utterances", [])
                    if utterance.get("speaker_id") is not None
                }
            ),
            "resources": stage_rows["transcribe"]["resources"],
            "gpu_time_seconds": None,
            "peak_vram_mb": None,
            "instrumentation_status": "gpu_metrics_unavailable",
        },
        translation={
            **translation_metrics,
            "wall_seconds": manifest.stages["localize"].duration_seconds,
            "resources": stage_rows["localize"]["resources"],
            "validation_failures": sum(
                item.failed_attempts
                for item in status.work_items.get("translation_batches", [])
            ),
        },
        synthesis={
            **synthesis_metrics,
            "wall_seconds": manifest.stages["synthesize"].duration_seconds,
            "resources": stage_rows["synthesize"]["resources"],
        },
        timing={
            **duration_metrics,
            "median_absolute_error_ms": (
                round(statistics.median(errors_ms), 3) if errors_ms else None
            ),
            "p95_absolute_error_ms": _percentile(errors_ms, 95),
            "p95_absolute_error_ratio": _percentile(ratios, 95),
        },
        rendering=(
            render_validation
            or {
                "status": (
                    "invalid_or_unreadable"
                    if render_report_declared
                    else "not_measured"
                )
            }
        ),
        stages={
            "items": stage_rows,
            "total_wall_seconds": round(total_wall, 6),
            "total_attempts": sum(item.attempt_count for item in manifest.stages.values()),
            "failed_attempts": sum(
                sum(attempt.status == StageStatus.FAILED for attempt in item.attempts)
                for item in manifest.stages.values()
            ),
        },
        resources={
            **status.resources,
            "gpu_time_seconds": None,
            "peak_vram_mb": None,
            "gpu_instrumentation_status": "not_measured",
        },
        cost={
            "reported_external_usd": sum(reported_costs) if reported_costs else None,
            "gpu_usd": None,
            "storage_usd": None,
            "total_usd": None,
            "per_input_minute_usd": None,
            "status": "incomplete_without_gpu_and_storage_pricing",
        },
        storage={
            "bytes": storage_bytes,
            "bytes_per_input_minute": round(storage_bytes / duration_minutes, 3)
            if duration_minutes > 0
            else None,
        },
        integrity={
            "source_utterance_count": len(source_ids),
            "localized_utterance_count": len(localized_ids),
            "synthesized_utterance_count": len(synthesized_ids),
            "missing_utterance_ids": missing_ids,
            "duplicate_utterance_ids": duplicate_ids,
            "unknown_utterance_ids": unknown_ids,
            "voice_consistency": voice_consistency,
        },
        human_review=human,
        quality_gates=gates,
        release_gate_status=release_status,
        missing_evidence=missing_evidence,
    )


def _benchmark_inputs(
    *,
    manifest: RunManifest,
    run_directory: Path,
    human_review_path: Path | None,
) -> dict[str, Any]:
    relevant_outputs = {
        name: value
        for name, value in manifest.outputs.items()
        if not name.startswith("benchmark_") and not name.startswith("human_review_")
    }
    output_hashes = {}
    for name, value in sorted(relevant_outputs.items()):
        path = Path(value)
        if path.is_file():
            output_hashes[name] = sha256_file(path)
        else:
            output_hashes[name] = None
    return {
        "run_id": manifest.run_id,
        "source_range_ms": [manifest.source_start_ms, manifest.source_end_ms],
        "languages": [manifest.source_language, manifest.target_language],
        "status": manifest.status.value,
        "media": manifest.media.model_dump(mode="json") if manifest.media else None,
        "models": manifest.models,
        "stages": {
            name: {
                "status": record.status.value,
                "attempt_count": record.attempt_count,
                "duration_seconds": record.duration_seconds,
                "provider": record.provider,
                "model": record.model,
                "input_fingerprint": record.input_fingerprint,
                "cost_usd": record.cost_usd,
                "resources": record.resources.model_dump(mode="json") if record.resources else None,
                "attempt_statuses": [item.status.value for item in record.attempts],
                "attempt_error_classes": [item.error_class for item in record.attempts],
            }
            for name, record in manifest.stages.items()
        },
        "error_classes": [item.error_class for item in manifest.error_records],
        "output_sha256": output_hashes,
        "human_review_sha256": (
            sha256_file(human_review_path)
            if human_review_path is not None and human_review_path.is_file()
            else None
        ),
        "benchmark_contract": "phase12_v1",
    }


def _human_review_template(
    manifest: RunManifest, run_directory: Path
) -> dict[str, Any]:
    segments = _read_list(
        manifest.outputs.get("localized_segments")
        or manifest.outputs.get("translation_segments")
    )
    corrections = {
        str(item.get("utterance_id")): item
        for item in _read_list(manifest.outputs.get("duration_corrections"))
    }
    transcript = _read_json(manifest.outputs.get("transcript"))
    translation_context = _read_json(
        manifest.outputs.get("translation_context")
    )
    low_confidence = _low_confidence_ids(transcript)
    selected: dict[str, set[str]] = {}
    if segments:
        selected.setdefault(str(segments[0].get("segment_id")), set()).add("beginning")
        selected.setdefault(
            str(segments[len(segments) // 2].get("segment_id")), set()
        ).add("middle")
        selected.setdefault(str(segments[-1].get("segment_id")), set()).add("end")
    rates = [
        (
            len(str(item.get("source_text", "")))
            / max(0.001, (int(item.get("end_ms", 1)) - int(item.get("start_ms", 0))) / 1000),
            item,
        )
        for item in segments
        if int(item.get("end_ms", 0)) > int(item.get("start_ms", 0))
    ]
    if rates:
        slow = min(rates, key=lambda pair: pair[0])[1]
        fast = max(rates, key=lambda pair: pair[0])[1]
        selected.setdefault(str(slow.get("segment_id")), set()).add("slow_speech")
        selected.setdefault(str(fast.get("segment_id")), set()).add("fast_speech")
    seen_speakers: set[str] = set()
    selected_technical = False
    selected_low_confidence = False
    selected_rewrite = False
    selected_name = False
    named_entities = translation_context.get("named_entities", [])
    for item in segments:
        identifier = str(item.get("segment_id"))
        speaker = str(item.get("speaker_id") or "speaker_unknown")
        if speaker not in seen_speakers:
            selected.setdefault(identifier, set()).add(f"speaker:{speaker}")
            seen_speakers.add(speaker)
        if item.get("glossary_terms") and not selected_technical:
            selected.setdefault(identifier, set()).add("technical_terms")
            selected_technical = True
        if identifier in low_confidence and not selected_low_confidence:
            selected.setdefault(identifier, set()).add("low_confidence_asr")
            selected_low_confidence = True
        correction = corrections.get(identifier, {})
        if correction.get("rewritten") and not selected_rewrite:
            selected.setdefault(identifier, set()).add("timing_rewrite")
            selected_rewrite = True
        if named_entities and not selected_name:
            haystack = (
                str(item.get("source_text", ""))
                + " "
                + str(item.get("target_text", ""))
            ).casefold()
            if any(
                str(entity.get("source", "")).casefold() in haystack
                or str(entity.get("target", "")).casefold() in haystack
                for entity in named_entities
                if isinstance(entity, dict)
            ):
                selected.setdefault(identifier, set()).add("names")
                selected_name = True
    if named_entities and not selected_name and segments:
        selected.setdefault(str(segments[0].get("segment_id")), set()).add(
            "names"
        )
    by_id = {str(item.get("segment_id")): item for item in segments}
    return {
        "schema_version": 1,
        "reviewer": None,
        "reviewed_at": None,
        "source_conditions": {
            "noise_present": None,
            "music_present": None,
            "overlap_present": None,
        },
        "instructions": (
            "Fill every null, add noise/music/overlap tags when declared present, "
            "score 1–5, and record critical defects separately."
        ),
        "available_utterance_ids": sorted(by_id),
        "samples": [
            {
                "utterance_id": identifier,
                "start_ms": by_id.get(identifier, {}).get("start_ms"),
                "end_ms": by_id.get(identifier, {}).get("end_ms"),
                "speaker_id": by_id.get(identifier, {}).get("speaker_id"),
                "coverage_tags": sorted(tags),
                "scores": {
                    "semantic_adequacy": None,
                    "naturalness": None,
                    "pronunciation": None,
                    "timing_quality": None,
                    "speaker_consistency": None,
                    "overall_usability": None,
                },
                "notes": None,
            }
            for identifier, tags in sorted(
                selected.items(),
                key=lambda pair: by_id.get(pair[0], {}).get("start_ms", 0),
            )
            if identifier and identifier != "None"
        ],
        "critical_defects": [],
    }


def _human_review_summary(
    review: HumanReviewSubmission | None, template: dict[str, Any]
) -> dict[str, Any]:
    required_tags = {
        tag
        for sample in template.get("samples", [])
        for tag in sample.get("coverage_tags", [])
    }
    if review is None:
        return {
            "status": "not_measured",
            "sample_count": 0,
            "required_tags": sorted(required_tags),
            "covered_tags": [],
            "missing_coverage_tags": sorted(required_tags),
            "median_semantic_adequacy": None,
            "median_scores": None,
            "critical_defect_count": 0,
            "critical_mistranslation_count": None,
        }
    available_ids = set(template.get("available_utterance_ids", []))
    unknown = sorted(
        {sample.utterance_id for sample in review.samples} - available_ids
    )
    if unknown:
        raise BenchmarkError(
            "Human review contains unknown utterance IDs: " + ", ".join(unknown)
        )
    condition_tags = {
        "noise": review.source_conditions.noise_present,
        "music": review.source_conditions.music_present,
        "overlap": review.source_conditions.overlap_present,
    }
    required_tags.update(tag for tag, present in condition_tags.items() if present)
    covered = {
        tag for sample in review.samples for tag in sample.coverage_tags
    }
    score_names = list(HumanReviewScores.model_fields)
    medians = {
        name: statistics.median(
            getattr(sample.scores, name) for sample in review.samples
        )
        for name in score_names
    }
    defect_unknown = sorted(
        {
            item.utterance_id
            for item in review.critical_defects
            if item.utterance_id is not None
        }
        - available_ids
    )
    if defect_unknown:
        raise BenchmarkError(
            "Human review defects contain unknown utterance IDs: "
            + ", ".join(defect_unknown)
        )
    critical = [
        item for item in review.critical_defects if item.severity == "critical"
    ]
    mistranslations = [
        item
        for item in critical
        if item.category in {"mistranslation", "semantic_omission", "incorrect_name"}
    ]
    return {
        "status": "completed",
        "reviewer": review.reviewer,
        "reviewed_at": review.reviewed_at.isoformat(),
        "sample_count": len(review.samples),
        "required_tags": sorted(required_tags),
        "covered_tags": sorted(covered),
        "missing_coverage_tags": sorted(required_tags - covered),
        "median_semantic_adequacy": medians["semantic_adequacy"],
        "median_scores": medians,
        "critical_defect_count": len(critical),
        "critical_mistranslation_count": len(mistranslations),
    }


def _load_human_review(path: Path | None) -> HumanReviewSubmission | None:
    if path is None:
        return None
    if not path.is_file():
        raise BenchmarkError(f"Human-review submission is missing: {path}")
    try:
        return HumanReviewSubmission.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError) as error:
        raise BenchmarkError(
            f"Human-review submission is invalid: {path}: {error}"
        ) from error


def _load_render_report(path: str | None) -> RenderReport | None:
    if not path:
        return None
    try:
        return RenderReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def _gate(
    gate_id: str,
    description: str,
    passed: bool,
    observed: Any,
    required: Any,
    *,
    measured: bool = True,
) -> QualityGate:
    return QualityGate(
        gate_id=gate_id,
        description=description,
        status=(
            EvidenceStatus.NOT_MEASURED
            if not measured
            else EvidenceStatus.PASSED if passed else EvidenceStatus.FAILED
        ),
        observed=observed,
        required=required,
    )


def _optional_numeric_gate(
    gate_id: str,
    description: str,
    observed: Any,
    *,
    minimum: float,
    required: Any,
) -> QualityGate:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return QualityGate(
            gate_id=gate_id,
            description=description,
            status=EvidenceStatus.NOT_MEASURED,
            observed=None,
            required=required,
        )
    return _gate(gate_id, description, value >= minimum, value, required)


def _voice_consistency(synthesized: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, set[str]] = {}
    for item in synthesized:
        speaker = str(item.get("speaker_id") or "speaker_unknown")
        voice = item.get("voice_id") or item.get("reference_id")
        if voice is not None:
            grouped.setdefault(speaker, set()).add(str(voice))
    inconsistent = sorted(
        speaker for speaker, voices in grouped.items() if len(voices) != 1
    )
    return {
        "speaker_count": len(grouped),
        "consistent_speaker_count": len(grouped) - len(inconsistent),
        "inconsistent_speaker_count": len(inconsistent),
        "inconsistent_speakers": inconsistent,
        "percent": (
            round((len(grouped) - len(inconsistent)) / len(grouped) * 100, 3)
            if grouped
            else None
        ),
    }


def _has_interruption_evidence(status: RunStatusDocument) -> bool:
    stage_attempts = [
        attempt
        for detail in status.stage_details.values()
        for attempt in detail.get("attempts", [])
    ]
    work_attempts = [
        attempt
        for group in status.work_items.values()
        for item in group
        for attempt in item.attempts
    ]
    return any(
        item.get("error_class") in {"interrupted", "lease_expired"}
        for item in [*stage_attempts, *work_attempts]
    )


def _artifact_reuse_evidence(
    translation_metrics: dict[str, Any], synthesis_metrics: dict[str, Any]
) -> bool:
    return (
        int(translation_metrics.get("reused_batches", 0)) > 0
        or int(synthesis_metrics.get("reused_utterances", 0)) > 0
    )


def _transcript_confidences(transcript: dict[str, Any]) -> list[float]:
    return [
        float(word["confidence"])
        for utterance in transcript.get("utterances", [])
        for word in utterance.get("words", [])
        if word.get("confidence") is not None
    ]


def _low_confidence_ids(transcript: dict[str, Any]) -> set[str]:
    result = set()
    for utterance in transcript.get("utterances", []):
        values = [
            float(item["confidence"])
            for item in utterance.get("words", [])
            if item.get("confidence") is not None
        ]
        if values and statistics.fmean(values) < 0.75:
            result.add(str(utterance.get("utterance_id")))
    return result


def _percentile(values: list[float] | list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _ids(payload: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("utterance_id") or item.get("segment_id"))
        for item in payload
        if item.get("utterance_id") is not None or item.get("segment_id") is not None
    ]


def _duplicates(items: list[str]) -> list[str]:
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)


def _run_storage_bytes(run_directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in run_directory.rglob("*")
        if path.is_file() and "benchmark" not in path.relative_to(run_directory).parts
    )


def _resolve_source(value: str, run_directory: Path) -> Path | None:
    source = Path(value)
    if source.is_absolute():
        return source
    inside = run_directory / source
    return inside if inside.exists() else None


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _read_list(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    except (OSError, ValueError, TypeError):
        return []


def _find_reusable_benchmark(
    *,
    directory: Path,
    stem: str,
    expected_inputs: dict[str, Any],
    root: Path,
) -> tuple[BenchmarkReport, BenchmarkArtifacts] | None:
    for metadata_path in sorted(
        directory.glob(f"{stem}-r*.json.meta.json"), reverse=True
    ):
        json_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json")
        )
        label = json_path.stem.rsplit("-", 1)[-1]
        markdown_path = directory / f"{stem}-{label}.md"
        template_path = directory / f"human-review-{stem[10:26]}-{label}.json"
        paths = {
            "benchmark_json": json_path,
            "benchmark_markdown": markdown_path,
            "human_review_template": template_path,
        }
        if not all(
            _verified(
                path=path,
                metadata_path=_metadata_path(path),
                expected_inputs=_artifact_inputs(expected_inputs, kind),
                root=root,
            )
            for kind, path in paths.items()
        ):
            continue
        try:
            report = BenchmarkReport.model_validate_json(
                json_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, ValidationError):
            continue
        if report.benchmark_fingerprint != fingerprint_inputs(expected_inputs):
            continue
        artifacts = BenchmarkArtifacts(
            benchmark_json=relative_artifact_path(json_path, root),
            benchmark_json_metadata=relative_artifact_path(metadata_path, root),
            benchmark_markdown=relative_artifact_path(markdown_path, root),
            benchmark_markdown_metadata=relative_artifact_path(
                _metadata_path(markdown_path), root
            ),
            human_review_template=relative_artifact_path(template_path, root),
            human_review_template_metadata=relative_artifact_path(
                _metadata_path(template_path), root
            ),
        )
        return report, artifacts
    return None


def _verified(
    *,
    path: Path,
    metadata_path: Path,
    expected_inputs: dict[str, Any],
    root: Path,
) -> bool:
    try:
        metadata = ArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        return (
            (root / metadata.path).resolve() == path.resolve()
            and verify_artifact(
                metadata, expected_inputs=expected_inputs, root=root
            ).valid
        )
    except (OSError, ValueError, ValidationError):
        return False


def _publish_benchmark_outputs(
    run_directory: Path, artifacts: BenchmarkArtifacts
) -> None:
    outputs = artifacts.as_outputs(run_directory)

    def apply(manifest: RunManifest) -> None:
        if all(manifest.outputs.get(name) == value for name, value in outputs.items()):
            raise MutationAborted
        manifest.outputs.update(outputs)

    mutate_manifest(run_directory, apply)


def _artifact_inputs(inputs: dict[str, Any], kind: str) -> dict[str, Any]:
    return {"benchmark": inputs, "kind": kind}


def _next_revision(directory: Path, stem: str) -> int:
    revisions = []
    for path in directory.glob(f"{stem}-r*.json"):
        if path.name.endswith(".meta.json"):
            continue
        raw = path.stem.rsplit("-r", 1)[-1]
        try:
            revisions.append(int(raw))
        except ValueError:
            continue
    return max(revisions, default=0) + 1


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
    )


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _display(
    value: Any, *, prefix: str = "", suffix: str = ""
) -> str:
    return "not measured" if value is None else f"{prefix}{value}{suffix}"


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True).replace("|", "\\|")
    return str(value).replace("|", "\\|")

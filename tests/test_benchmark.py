import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dub_mvp.artifacts import ArtifactMetadata
from dub_mvp.benchmark import (
    BenchmarkError,
    BenchmarkReport,
    EvidenceStatus,
    build_benchmark,
)
from dub_mvp.cli import app
from dub_mvp.manifest import (
    MediaMetadata,
    ResourceUsage,
    RunManifest,
    RunStatus,
    StageAttempt,
    StageStatus,
)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def benchmark_run(tmp_path: Path) -> Path:
    run = tmp_path / "benchmark-run"
    source = run / "input" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-video-bytes")
    identifiers = ["utt_0001", "utt_0002"]
    segments = [
        {
            "segment_id": identifier,
            "start_ms": index * 1000,
            "end_ms": index * 1000 + 900,
            "duration_budget_ms": 900,
            "speaker_id": f"speaker_{index + 1}",
            "source_text": f"API name {index}",
            "target_text": f"API लक्ष्य {index}",
            "glossary_terms": ["API"] if index == 0 else [],
        }
        for index, identifier in enumerate(identifiers)
    ]
    localized = [
        {**item, "target_text_revision": 1} for item in segments
    ]
    synthesized = [
        {
            **item,
            "target_text_revision": 1,
            "voice_id": f"voice_{index + 1}",
            "reference_id": f"voice_{index + 1}",
        }
        for index, item in enumerate(segments)
    ]
    corrections = [
        {
            "utterance_id": identifiers[0],
            "duration_error_ms": 50,
            "duration_ratio": 950 / 900,
            "within_primary_tolerance": True,
            "within_hard_tolerance": True,
            "rewritten": False,
        },
        {
            "utterance_id": identifiers[1],
            "duration_error_ms": -100,
            "duration_ratio": 800 / 900,
            "within_primary_tolerance": True,
            "within_hard_tolerance": True,
            "rewritten": True,
        },
    ]
    transcript = {
        "utterances": [
            {
                "utterance_id": identifiers[0],
                "words": [{"text": "API", "confidence": 0.95}],
            },
            {
                "utterance_id": identifiers[1],
                "words": [{"text": "name", "confidence": 0.6}],
            },
        ]
    }
    outputs = {
        "transcript": str(_write_json(run / "metadata" / "transcript.json", transcript)),
        "translation_segments": str(
            _write_json(run / "utterances" / "translation.json", segments)
        ),
        "localized_segments": str(
            _write_json(run / "translation" / "localized.json", localized)
        ),
        "synthesized_segments": str(
            _write_json(run / "speech" / "synthesized.json", synthesized)
        ),
        "duration_corrections": str(
            _write_json(run / "speech" / "duration.json", corrections)
        ),
        "duration_metrics": str(
            _write_json(
                run / "speech" / "duration-metrics.json",
                {
                    "utterance_count": 2,
                    "within_primary_percent": 100.0,
                    "within_hard_percent": 100.0,
                    "unresolved_count": 0,
                    "rewrite_count": 1,
                    "human_review_required_count": 1,
                },
            )
        ),
        "translation_metrics": str(
            _write_json(
                run / "translation" / "metrics.json",
                {
                    "batch_count": 1,
                    "provider_calls": 0,
                    "reused_batches": 1,
                    "attempt_count": 2,
                    "failed_attempts": 1,
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "provider_latency_seconds": 1.2,
                    "cost_usd": 0.02,
                },
            )
        ),
        "synthesis_metrics": str(
            _write_json(
                run / "speech" / "metrics.json",
                {
                    "provider_calls": 0,
                    "reused_utterances": 2,
                    "attempt_count": 3,
                    "failed_attempts": 1,
                    "provider_latency_seconds": 1.5,
                    "generated_duration_ms": 1800,
                },
            )
        ),
        "translation_context": str(
            _write_json(
                run / "translation" / "context.json",
                {"named_entities": [{"source": "name", "target": "नाम"}]},
            )
        ),
    }
    render_report = {
        "schema_version": 1,
        "configuration_fingerprint": "f" * 64,
        "composition_mode": "clean_replacement",
        "source": {
            "path": "working/source.mp4",
            "duration_ms": 1_800_000,
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "frame_rate": "30/1",
                }
            ],
        },
        "dubbed_audio": {
            "path": "working/dubbed.wav",
            "duration_ms": 1_800_000,
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate_hz": 48000,
                    "channels": 2,
                }
            ],
        },
        "output": {
            "path": "outputs/dubbed.mp4",
            "duration_ms": 1_800_000,
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "frame_rate": "30/1",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate_hz": 48000,
                    "channels": 2,
                },
            ],
        },
        "validation": {
            "expected_duration_ms": 1_800_000,
            "output_duration_ms": 1_800_000,
            "duration_error_ms": 0,
            "duration_within_tolerance": True,
            "audio_duration_ms": 1_800_000,
            "audio_sample_rate_hz": 48000,
            "audio_channels": 2,
            "decoded_peak_dbfs": -1.0,
            "clipping_detected": False,
            "full_decode_succeeded": True,
            "video_stream_copied": True,
            "source_video_codec": "h264",
            "output_video_codec": "h264",
            "source_width": 1920,
            "source_height": 1080,
            "output_width": 1920,
            "output_height": 1080,
            "source_frame_rate": "30/1",
            "output_frame_rate": "30/1",
            "source_start_drift_ms": 0,
            "passed": True,
        },
        "artifacts": {},
        "commands_path": "render/commands.json",
        "command_attempt_count": 1,
        "completed_at": "2026-08-13T12:00:00Z",
    }
    outputs["render_report"] = str(
        _write_json(run / "render" / "report.json", render_report)
    )
    manifest = RunManifest(
        run_id="benchmark-run",
        source_path="input/source.mp4",
        source_start_ms=0,
        source_end_ms=1_800_000,
        status=RunStatus.RENDERED,
        media=MediaMetadata(
            duration_seconds=1800,
            format_name="mov,mp4",
            video_codec="h264",
            width=1920,
            height=1080,
            frame_rate="30/1",
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate=48000,
        ),
    )
    manifest.outputs.update(outputs)
    for index, name in enumerate(manifest.stages):
        record = manifest.stages[name]
        record.status = StageStatus.COMPLETED
        record.attempt_count = 1
        record.duration_seconds = float(index + 1)
        record.resources = ResourceUsage(
            wall_seconds=float(index + 1),
            cpu_user_seconds=0.5,
            cpu_system_seconds=0.1,
            max_rss_mb=256,
        )
        record.attempts.append(
            StageAttempt(
                attempt_number=1,
                status=StageStatus.COMPLETED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )
    # Durable evidence that a prior attempt was interrupted and then recovered.
    transcribe = manifest.stages["transcribe"]
    transcribe.attempt_count = 2
    transcribe.attempts = [
        StageAttempt(
            attempt_number=1,
            status=StageStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            error_class="interrupted",
            error="worker process ended",
        ),
        StageAttempt(
            attempt_number=2,
            status=StageStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        ),
    ]
    manifest.stages["localize"].cost_usd = 0.02
    manifest.save(run)
    return run


def test_benchmark_writes_verified_json_markdown_and_review_template(
    tmp_path: Path,
) -> None:
    run = benchmark_run(tmp_path)

    report, artifacts = build_benchmark(run)

    outputs = artifacts.as_outputs(run)
    assert report.benchmark_scope["long_form_qualified"]
    assert report.timing["median_absolute_error_ms"] == 75
    assert report.timing["p95_absolute_error_ms"] == 97.5
    assert report.integrity["missing_utterance_ids"] == []
    assert report.integrity["voice_consistency"]["percent"] == 100
    assert report.transcription["word_confidence_mean"] == 0.775
    assert report.rendering["passed"] is True
    assert report.release_gate_status == EvidenceStatus.NOT_MEASURED
    assert "Complete runtime, GPU/resource, retry, and cost reporting" in report.missing_evidence
    assert Path(outputs["benchmark_json"]).is_file()
    assert Path(outputs["benchmark_markdown"]).is_file()
    markdown = Path(outputs["benchmark_markdown"]).read_text(encoding="utf-8")
    assert "Release gate: **not_measured**" in markdown
    assert "never treats unavailable GPU" in markdown
    metadata = ArtifactMetadata.model_validate_json(
        Path(outputs["benchmark_json_metadata"]).read_text(encoding="utf-8")
    )
    assert not Path(metadata.path).is_absolute()
    template = json.loads(
        Path(outputs["human_review_template"]).read_text(encoding="utf-8")
    )
    tags = {tag for item in template["samples"] for tag in item["coverage_tags"]}
    assert {"beginning", "middle", "end", "fast_speech", "slow_speech"} <= tags
    assert {"technical_terms", "timing_rewrite", "low_confidence_asr", "names"} <= tags
    assert {"speaker:speaker_1", "speaker:speaker_2"} <= tags
    loaded = RunManifest.load(run)
    assert loaded.outputs["benchmark_json"] == outputs["benchmark_json"]


def test_benchmark_reuses_verified_report_without_manifest_churn(
    tmp_path: Path,
) -> None:
    run = benchmark_run(tmp_path)
    first_report, first = build_benchmark(run)
    revision = RunManifest.load(run).revision

    second_report, second = build_benchmark(run)

    assert second == first
    assert second_report.generated_at == first_report.generated_at
    assert RunManifest.load(run).revision == revision


def test_corrupt_benchmark_is_rebuilt_as_a_new_revision(tmp_path: Path) -> None:
    run = benchmark_run(tmp_path)
    _, first = build_benchmark(run)
    first_json = Path(first.as_outputs(run)["benchmark_json"])
    first_json.write_text("corrupt", encoding="utf-8")

    _, second = build_benchmark(run)

    second_json = Path(second.as_outputs(run)["benchmark_json"])
    assert second_json != first_json
    assert second_json.name.endswith("r0002.json")
    assert first_json.read_text(encoding="utf-8") == "corrupt"


def test_completed_human_review_is_aggregated_without_hiding_critical_defects(
    tmp_path: Path,
) -> None:
    run = benchmark_run(tmp_path)
    _, first = build_benchmark(run)
    template_path = Path(first.as_outputs(run)["human_review_template"])
    review = json.loads(template_path.read_text(encoding="utf-8"))
    review["reviewer"] = "Reviewer A"
    review["reviewed_at"] = "2026-08-13T13:00:00Z"
    review["source_conditions"] = {
        "noise_present": False,
        "music_present": False,
        "overlap_present": False,
    }
    review.pop("instructions")
    review.pop("available_utterance_ids")
    for sample in review["samples"]:
        sample["scores"] = {
            "semantic_adequacy": 4,
            "naturalness": 4,
            "pronunciation": 4,
            "timing_quality": 4,
            "speaker_consistency": 5,
            "overall_usability": 4,
        }
    review["critical_defects"] = [
        {
            "category": "pronunciation",
            "severity": "critical",
            "description": "A required name was pronounced incorrectly.",
            "utterance_id": "utt_0001",
        }
    ]
    review_path = _write_json(tmp_path / "human-review.json", review)

    report, _ = build_benchmark(run, human_review_path=review_path)

    assert report.human_review["status"] == "completed"
    assert report.human_review["median_semantic_adequacy"] == 4
    assert report.human_review["missing_coverage_tags"] == []
    assert report.human_review["critical_defect_count"] == 1
    assert report.human_review["critical_mistranslation_count"] == 0
    assert next(
        gate for gate in report.quality_gates if gate.gate_id == "semantic_adequacy"
    ).status == EvidenceStatus.PASSED


def test_human_review_rejects_unknown_utterance_id(tmp_path: Path) -> None:
    run = benchmark_run(tmp_path)
    review = {
        "reviewer": "Reviewer",
        "reviewed_at": "2026-08-13T13:00:00Z",
        "source_conditions": {
            "noise_present": False,
            "music_present": False,
            "overlap_present": False,
        },
        "samples": [
            {
                "utterance_id": "unknown",
                "coverage_tags": ["beginning"],
                "scores": {
                    "semantic_adequacy": 5,
                    "naturalness": 5,
                    "pronunciation": 5,
                    "timing_quality": 5,
                    "speaker_consistency": 5,
                    "overall_usability": 5,
                },
            }
        ],
    }
    review_path = _write_json(tmp_path / "review.json", review)

    with pytest.raises(BenchmarkError, match="unknown utterance IDs"):
        build_benchmark(run, human_review_path=review_path)


def test_benchmark_cli_reports_outputs_and_incomplete_gate(tmp_path: Path) -> None:
    run = benchmark_run(tmp_path)

    result = CliRunner().invoke(app, ["benchmark", str(run)])

    assert result.exit_code == 0, result.output
    assert "Release gate: not_measured" in result.output
    assert "Human review template:" in result.output
    report = BenchmarkReport.model_validate_json(
        Path(RunManifest.load(run).outputs["benchmark_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert report.resources["gpu_instrumentation_status"] == "not_measured"
    assert report.cost["gpu_usd"] is None

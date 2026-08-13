import json
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dub_mvp.cli import app
from dub_mvp.manifest import (
    ResourceUsage,
    RunManifest,
    RunStatus,
    StageAttempt,
    StageStatus,
    append_stage_event,
    complete_stage,
    fail_stage,
    mutate_manifest,
    redact_sensitive_text,
)
from dub_mvp.observability import build_run_status, load_run_events
from dub_mvp.ui import build_customer_run_payload


def observed_run(tmp_path: Path) -> Path:
    run = tmp_path / "observed-run"
    translation = run / "translation" / "batches"
    speech = run / "speech" / "utterances" / "u-one"
    translation.mkdir(parents=True)
    speech.mkdir(parents=True)
    segments = run / "segments.json"
    localized = run / "localized.json"
    synthesized = run / "synthesized.json"
    for path in (segments, localized, synthesized):
        path.write_text('[{"segment_id":"u-one"}]', encoding="utf-8")
    (translation / "batch_0001.attempts.json").write_text(
        json.dumps(
            [
                {
                    "attempt_number": 1,
                    "batch_id": "batch_0001",
                    "status": "failed",
                    "started_at": "2026-08-12T12:00:00Z",
                    "completed_at": "2026-08-12T12:00:01Z",
                    "latency_seconds": 1.0,
                    "provider": "fixture",
                    "model": "translator-a",
                    "cost_usd": 0.01,
                    "error": "timeout token=secret-value",
                },
                {
                    "attempt_number": 2,
                    "batch_id": "batch_0001",
                    "status": "completed",
                    "started_at": "2026-08-12T12:01:00Z",
                    "completed_at": "2026-08-12T12:01:02Z",
                    "latency_seconds": 2.0,
                    "provider": "fixture",
                    "model": "translator-a",
                    "cost_usd": 0.02,
                },
            ]
        ),
        encoding="utf-8",
    )
    (speech / "tts.attempts.json").write_text(
        json.dumps(
            [
                {
                    "attempt_number": 1,
                    "utterance_id": "u-one",
                    "status": "completed",
                    "started_at": "2026-08-12T12:02:00Z",
                    "completed_at": "2026-08-12T12:02:01Z",
                    "latency_seconds": 1.0,
                    "provider": "fixture-tts",
                    "model": "tts-a",
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = RunManifest(
        run_id="observed-run",
        source_path="input/source.mp4",
        source_start_ms=0,
        source_end_ms=5000,
        status=RunStatus.LOCALIZED,
    )
    manifest.outputs.update(
        {
            "translation_segments": str(segments),
            "localized_segments": str(localized),
            "synthesized_segments": str(synthesized),
        }
    )
    record = manifest.stages["localize"]
    record.status = StageStatus.COMPLETED
    record.attempt_count = 1
    record.provider = "fixture"
    record.model = "translator-a"
    record.input_fingerprint = "a" * 64
    record.duration_seconds = 3.0
    record.cost_usd = 0.03
    record.resources = ResourceUsage(
        wall_seconds=3.0,
        cpu_user_seconds=1.5,
        cpu_system_seconds=0.25,
        max_rss_mb=256.0,
    )
    record.attempts.append(
        StageAttempt(
            attempt_number=1,
            status=StageStatus.COMPLETED,
            started_at=manifest.created_at,
            completed_at=manifest.created_at,
        )
    )
    append_stage_event(
        record,
        at=manifest.created_at,
        event="completed",
        from_status=StageStatus.RUNNING,
        to_status=StageStatus.COMPLETED,
    )
    manifest.timings_seconds["localize"] = 3.0
    manifest.save(run)
    return run


def test_status_explains_stage_work_item_cost_time_and_resources(
    tmp_path: Path,
) -> None:
    run = observed_run(tmp_path)

    status = build_run_status(run)

    assert status.stage_details["localize"]["attempt_count"] == 1
    assert status.stage_details["localize"]["provider"] == "fixture"
    assert status.progress["utterances"] == {
        "total": 1,
        "localized": 1,
        "synthesized": 1,
    }
    assert status.progress["attempts"] == {
        "stage": 1,
        "work_item": 3,
        "total": 4,
    }
    batch = status.work_items["translation_batches"][0]
    assert batch.work_item_id == "batch_0001"
    assert batch.attempt_count == 2
    assert batch.failed_attempts == 1
    assert batch.cost_usd == 0.03
    assert "secret-value" not in json.dumps(batch.model_dump(mode="json"))
    assert status.cost["reported_usd"] == 0.03
    assert status.resources["peak_rss_mb"] == 256.0
    assert status.timings_seconds == {"localize": 3.0}
    assert all(
        event["manifest_revision"] == status.manifest_revision
        for event in status.recent_events
    )


def test_cli_and_web_use_the_same_durable_status_document(tmp_path: Path) -> None:
    run = observed_run(tmp_path)
    expected = build_run_status(run).model_dump(mode="json")

    cli = CliRunner().invoke(app, ["status", str(run)])
    web = build_customer_run_payload(run)

    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == expected
    assert web["summary"] == expected


def test_structured_errors_and_event_log_redact_credentials(
    tmp_path: Path,
) -> None:
    run = tmp_path / "redacted-run"
    manifest = RunManifest(
        run_id="redacted-run",
        source_path="input/source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
        status=RunStatus.RUNNING,
    )
    record = manifest.stages["transcribe"]
    record.status = StageStatus.RUNNING
    record.attempt_count = 1
    record.attempts.append(
        StageAttempt(
            attempt_number=1,
            status=StageStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
    )
    manifest.save(run)

    fail_stage(
        run,
        "transcribe",
        error="request failed token=secret-value sk-1234567890abcdef",
        error_class="ProviderError",
        retryable=False,
    )

    loaded = RunManifest.load(run)
    status = build_run_status(run)
    event_text = (run / status.event_log).read_text(encoding="utf-8")
    serialized = json.dumps(status.model_dump(mode="json"))
    assert loaded.error_records[0].error_class == "ProviderError"
    assert loaded.error_records[0].stage == "transcribe"
    assert "secret-value" not in serialized
    assert "1234567890abcdef" not in serialized
    assert "secret-value" not in event_text
    assert "[REDACTED]" in event_text
    assert "secret-value" not in (run / "manifest.json").read_text(
        encoding="utf-8"
    )


def test_corrupt_event_projection_is_rebuilt_from_manifest(tmp_path: Path) -> None:
    run = observed_run(tmp_path)
    event_path = run / "events" / "run-events.jsonl"
    event_path.write_text("not-json\n", encoding="utf-8")

    events = load_run_events(run)

    assert events[0]["event"] == "run_created"
    assert events[-1]["event"] == "completed"
    assert [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ] == events


def test_status_survives_an_unwritable_event_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = observed_run(tmp_path)
    (run / "events" / "run-events.jsonl").write_text(
        "corrupt\n", encoding="utf-8"
    )

    def fail_projection(*_args, **_kwargs):
        raise PermissionError("read-only event storage")

    monkeypatch.setattr(
        "dub_mvp.observability.write_run_event_log", fail_projection
    )

    status = build_run_status(run)

    assert status.run_id == "observed-run"
    assert status.recent_events[-1]["event"] == "completed"


def append_process_event(run_directory: str, index: int) -> None:
    run = Path(run_directory)

    def apply(manifest: RunManifest) -> None:
        append_stage_event(
            manifest.stages["ingest"],
            at=datetime.now(timezone.utc),
            event=f"process_{index}",
        )

    mutate_manifest(run, apply)


def test_concurrent_writers_leave_valid_complete_jsonl_projection(
    tmp_path: Path,
) -> None:
    run = tmp_path / "concurrent-run"
    RunManifest(
        run_id="concurrent-run",
        source_path="input/source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    ).save(run)
    processes = [
        Process(target=append_process_event, args=(str(run), index))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    events = load_run_events(run)

    assert {event["event"] for event in events} == {
        "run_created",
        "process_0",
        "process_1",
        "process_2",
        "process_3",
    }
    assert [event["sequence"] for event in events] == list(range(1, 6))


REDACTION_CASES = [
    ("openai_key", "Auth failed for sk-proj-AbCdEf123456789xyz", "AbCdEf123456789xyz"),
    (
        "authorization_header",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc",
        "eyJhbGciOiJIUzI1NiJ9.abc",
    ),
    ("env_assignment", "OPENAI_API_KEY=sk-live-9988776655443322", "9988776655443322"),
    (
        "query_string",
        "POST https://api.example.com/v1?api_key=abc123def456",
        "abc123def456",
    ),
    # Provider SDKs put the request body in the exception text, so this is the
    # most likely way a key reaches a durable manifest.
    (
        "json_body_echo",
        'request failed: {"api_key": "abc123def456", "model": "gpt"}',
        "abc123def456",
    ),
    (
        "json_bearer_echo",
        '{"Authorization": "Bearer eyJhbGciOiJI.secret"}',
        "eyJhbGciOiJI.secret",
    ),
    (
        "url_basic_auth",
        "connect failed: https://admin:hunter2@db.internal:5432/runs",
        "hunter2",
    ),
    ("aws_access_key", "denied for AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    (
        "aws_secret",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCY",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCY",
    ),
    (
        "huggingface_token",
        "invalid token hf_QwErTyUiOpAsDfGhJkLzXcVbNm1234",
        "QwErTyUiOpAsDfGhJkLzXcVbNm1234",
    ),
]


@pytest.mark.parametrize(
    ("label", "text", "secret"),
    REDACTION_CASES,
    ids=[case[0] for case in REDACTION_CASES],
)
def test_credential_formats_are_redacted(
    label: str,
    text: str,
    secret: str,
) -> None:
    redacted = redact_sensitive_text(text)

    assert secret not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.parametrize(
    "text",
    [case[1] for case in REDACTION_CASES],
    ids=[case[0] for case in REDACTION_CASES],
)
def test_redaction_is_idempotent(text: str) -> None:
    # Errors are redacted on write and again when an old manifest is migrated
    # on read, so a second pass must not corrupt the message.
    once = redact_sensitive_text(text)

    assert redact_sensitive_text(once) == once


@pytest.mark.parametrize(
    "text",
    [
        "ffmpeg exited with code 1: no such file /tmp/source.wav",
        "GET https://api.openai.com/v1/responses returned 429",
        "Segment seg_0004 exceeds the translation batch character limit",
    ],
)
def test_redaction_leaves_ordinary_errors_readable(text: str) -> None:
    assert redact_sensitive_text(text) == text

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer

from dub_mvp.benchmark import BenchmarkError, build_benchmark
from dub_mvp.configuration import (
    ConfigurationError,
    build_configuration_snapshot,
    validate_release_language_pair,
    write_configuration_snapshot,
)
from dub_mvp.manifest import (
    RunManifest,
    RunStatus,
    StageAttempt,
    StageRecord,
    StageStatus,
    append_run_error,
    append_stage_event,
    redact_sensitive_text,
)
from dub_mvp.localize import (
    LocalizationError,
    LocalizationPipeline,
    TranslationMetrics,
    localization_outputs_reusable,
)
from dub_mvp.media import MediaIngestor, MediaToolError, media_duration_ms
from dub_mvp.observability import (
    ResourceSnapshot,
    build_run_status,
    capture_resource_snapshot,
    resources_since,
)
from dub_mvp.preflight import (
    PreflightProfile,
    build_preflight_report,
    report_to_json,
)
from dub_mvp.readiness import (
    DeploymentTarget,
    ReadinessStatus,
    assess_deployment_readiness,
    assess_language_expansion,
    assess_research_readiness,
    readiness_json,
)
from dub_mvp.render import (
    CompositionMode,
    RenderError,
    RenderPipeline,
    RenderPolicy,
    RenderReport,
    render_outputs_reusable,
)
from dub_mvp.retry import RetryError, RetryStage, retry_run
from dub_mvp.synthesize import (
    SynthesisError,
    SynthesisMetrics,
    SynthesisPipeline,
    synthesis_outputs_reusable,
)
from dub_mvp.timecode import parse_timecode_ms
from dub_mvp.transcribe import TranscriptionError, TranscriptionPipeline
from dub_mvp.ui import UiServer
from dub_mvp.utterances import UtteranceError, UtterancePipeline
from dub_mvp.webapp import ProductWebServer
from dub_mvp.worker import WorkerError, run_worker_loop, run_worker_once

app = typer.Typer(
    no_args_is_help=True,
    help="Build and inspect resumable video dubbing runs.",
)


@dataclass(frozen=True)
class CliStageMeasurement:
    started_at: datetime
    resources: ResourceSnapshot


def _start_cli_stage(
    manifest: RunManifest,
    stage_name: str,
) -> CliStageMeasurement:
    moment = datetime.now(timezone.utc)
    record = manifest.stages.setdefault(stage_name, StageRecord())
    previous_status = record.status
    record.status = StageStatus.RUNNING
    record.attempt_count += 1
    record.started_at = moment
    record.heartbeat_at = moment
    record.completed_at = None
    record.error = None
    record.error_class = None
    record.attempts.append(
        StageAttempt(
            attempt_number=record.attempt_count,
            status=StageStatus.RUNNING,
            started_at=moment,
            heartbeat_at=moment,
        )
    )
    append_stage_event(
        record,
        at=moment,
        event="started",
        from_status=previous_status,
        to_status=StageStatus.RUNNING,
    )
    manifest.status = RunStatus.RUNNING
    return CliStageMeasurement(
        started_at=moment,
        resources=capture_resource_snapshot(),
    )


def _complete_cli_stage(
    manifest: RunManifest,
    stage_name: str,
    measurement: CliStageMeasurement,
    *,
    run_status: RunStatus,
) -> None:
    moment = datetime.now(timezone.utc)
    record = manifest.stages[stage_name]
    duration = max(0.0, (moment - measurement.started_at).total_seconds())
    previous_status = record.status
    record.status = StageStatus.COMPLETED
    record.completed_at = moment
    record.heartbeat_at = moment
    record.duration_seconds = duration
    record.resources = resources_since(measurement.resources)
    if record.attempts:
        attempt = record.attempts[-1]
        attempt.status = StageStatus.COMPLETED
        attempt.completed_at = moment
        attempt.heartbeat_at = moment
    append_stage_event(
        record,
        at=moment,
        event="completed",
        from_status=previous_status,
        to_status=StageStatus.COMPLETED,
    )
    manifest.status = run_status
    manifest.timings_seconds[stage_name] = duration


def _fail_cli_stage(
    manifest: RunManifest,
    stage_name: str,
    measurement: CliStageMeasurement,
    error: Exception,
) -> str:
    moment = datetime.now(timezone.utc)
    record = manifest.stages[stage_name]
    duration = max(0.0, (moment - measurement.started_at).total_seconds())
    message = redact_sensitive_text(str(error))
    previous_status = record.status
    record.status = StageStatus.FAILED
    record.retryable = False
    record.error = message
    record.error_class = type(error).__name__
    record.completed_at = moment
    record.heartbeat_at = moment
    record.duration_seconds = duration
    record.resources = resources_since(measurement.resources)
    if record.attempts:
        attempt = record.attempts[-1]
        attempt.status = StageStatus.FAILED
        attempt.completed_at = moment
        attempt.heartbeat_at = moment
        attempt.error_class = type(error).__name__
        attempt.error = message
    append_stage_event(
        record,
        at=moment,
        event="failed",
        from_status=previous_status,
        to_status=StageStatus.FAILED,
        detail=message,
    )
    append_run_error(
        manifest,
        at=moment,
        stage=stage_name,
        error_class=type(error).__name__,
        message=message,
        retryable=False,
        terminal=True,
        attempt_number=record.attempt_count,
    )
    manifest.status = RunStatus.FAILED
    manifest.timings_seconds[stage_name] = duration
    return message


@app.command()
def ingest(
    input: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Source video file.",
    ),
    start: str = typer.Option("0", help="Start as seconds or HH:MM:SS."),
    end: str | None = typer.Option(
        None,
        help="Optional end as seconds or HH:MM:SS; defaults to full duration.",
    ),
    output: Path = typer.Option(
        Path("runs"),
        help="Parent directory for generated runs.",
    ),
    name: str | None = typer.Option(
        None,
        help="Optional human-readable run name.",
    ),
    source_language: str = typer.Option(
        "en",
        "--source-language",
        help="Release-enabled source language.",
    ),
    target_language: str = typer.Option(
        "hi",
        "--target-language",
        help="Release-enabled target language.",
    ),
) -> None:
    """Inspect and extract a source range into a new resumable run."""
    try:
        start_ms = parse_timecode_ms(start)
        ingestor = MediaIngestor()
        end_ms = (
            parse_timecode_ms(end)
            if end is not None
            else media_duration_ms(ingestor.inspect(input))
        )
    except (ValueError, MediaToolError) as error:
        raise typer.BadParameter(str(error)) from error
    if end_ms <= start_ms:
        raise typer.BadParameter("End time must be greater than start time.")
    try:
        source_language, target_language = validate_release_language_pair(
            source_language, target_language
        )
    except ConfigurationError as error:
        raise typer.BadParameter(str(error)) from error

    run_id = _new_run_id(name or input.stem)
    run_directory = output.expanduser().resolve() / run_id
    configuration_outputs = write_configuration_snapshot(
        build_configuration_snapshot(
            run_directory=run_directory,
            source_language=source_language,
            target_language=target_language,
        ),
        run_directory=run_directory,
    )
    manifest = RunManifest(
        run_id=run_id,
        source_path=str(input),
        source_start_ms=start_ms,
        source_end_ms=end_ms,
        source_language=source_language,
        target_language=target_language,
        outputs=configuration_outputs,
    )
    manifest.save(run_directory)

    stage = manifest.stages["ingest"]
    measurement = _start_cli_stage(manifest, "ingest")
    manifest.save(run_directory)

    try:
        metadata, outputs = ingestor.ingest(
            source=input,
            run_directory=run_directory,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    except MediaToolError as error:
        message = _fail_cli_stage(manifest, "ingest", measurement, error)
        manifest.save(run_directory)
        typer.echo(f"Ingest failed: {message}", err=True)
        typer.echo(f"Run manifest: {run_directory / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    _complete_cli_stage(
        manifest,
        "ingest",
        measurement,
        run_status=RunStatus.INGESTED,
    )
    stage.outputs = outputs
    stage.provider = "ffmpeg"
    manifest.media = metadata
    manifest.outputs.update(outputs)
    manifest.save(run_directory)

    typer.echo(f"Ingest complete: {run_directory}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def transcribe(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Existing run directory created by ingest.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-run transcription even when completed outputs exist.",
    ),
    model: str = typer.Option(
        "large-v3",
        help="WhisperX model name to record and load.",
    ),
) -> None:
    """Transcribe the ingested working audio into normalized segment JSON."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("transcribe", StageRecord())
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and _transcribe_outputs_exist(stage.outputs)
    ):
        typer.echo(f"Transcribe already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    audio_output = manifest.outputs.get("working_audio")
    if not audio_output:
        typer.echo("Transcribe requires a completed ingest stage.", err=True)
        raise typer.Exit(code=1)

    audio_path = Path(audio_output)
    measurement = _start_cli_stage(manifest, "transcribe")
    manifest.save(run)

    try:
        transcript, segments, outputs = TranscriptionPipeline(
            model_name=model,
        ).run(
            audio_path=audio_path,
            run_directory=run,
            language=manifest.source_language,
            duration_ms=manifest.duration_ms,
        )
    except TranscriptionError as error:
        message = _fail_cli_stage(
            manifest, "transcribe", measurement, error
        )
        manifest.save(run)
        typer.echo(f"Transcribe failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    _complete_cli_stage(
        manifest,
        "transcribe",
        measurement,
        run_status=RunStatus.TRANSCRIBED,
    )
    stage.outputs = outputs
    stage.provider = "whisperx"
    stage.model = transcript.model
    manifest.models["whisperx"] = transcript.model
    manifest.outputs.update(outputs)
    manifest.save(run)

    typer.echo(f"Transcribe complete: {run}")
    typer.echo(f"Segments: {len(segments)}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def segment(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Existing run directory with transcription outputs.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild dubbing utterances even when outputs exist.",
    ),
) -> None:
    """Create stable, speaker-aware dubbing utterances for localization."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("segment", StageRecord())
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and _segment_outputs_exist(stage.outputs)
    ):
        typer.echo(f"Segment already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    transcript_output = manifest.outputs.get("transcript")
    segments_output = manifest.outputs.get("segments")
    if not transcript_output or not segments_output:
        typer.echo("Segment requires completed transcription outputs.", err=True)
        raise typer.Exit(code=1)

    measurement = _start_cli_stage(manifest, "segment")
    manifest.save(run)

    try:
        artifact, translation_segments, outputs = UtterancePipeline().run(
            transcript_path=Path(transcript_output),
            segments_path=Path(segments_output),
            run_directory=run,
        )
    except UtteranceError as error:
        message = _fail_cli_stage(manifest, "segment", measurement, error)
        manifest.save(run)
        typer.echo(f"Segment failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    _complete_cli_stage(
        manifest,
        "segment",
        measurement,
        run_status=RunStatus.SEGMENTED,
    )
    stage.outputs = outputs
    stage.provider = "deterministic"
    manifest.outputs.update(outputs)
    manifest.save(run)

    typer.echo(f"Segment complete: {run}")
    typer.echo(f"Utterances: {len(artifact.utterances)}")
    typer.echo(f"Translation segments: {len(translation_segments)}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def localize(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Existing run directory with transcription outputs.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-run localization even when completed outputs exist.",
    ),
    glossary: Path | None = typer.Option(
        None,
        "--glossary",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional JSON glossary for technical terms.",
    ),
    context: Path | None = typer.Option(
        None,
        "--context",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional JSON tone, named-entity, and terminology context.",
    ),
    model: str = typer.Option(
        "gpt-5-mini",
        help="Translator model name to record and load.",
    ),
) -> None:
    """Localize source transcript segments into spoken Hindi text."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("localize", StageRecord())
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and localization_outputs_reusable(
            outputs=stage.outputs,
            segments_path=Path(
                manifest.outputs.get("translation_segments")
                or manifest.outputs.get("segments", "")
            ),
            run_directory=run,
            source_language=manifest.source_language,
            target_language=manifest.target_language,
            model_name=model,
            glossary_path=glossary,
            context_path=context,
        )
    ):
        typer.echo(f"Localize already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    segments_output = manifest.outputs.get(
        "translation_segments"
    ) or manifest.outputs.get("segments")
    if not segments_output:
        typer.echo("Localize requires completed transcription segments.", err=True)
        raise typer.Exit(code=1)

    measurement = _start_cli_stage(manifest, "localize")
    manifest.save(run)

    try:
        localized_segments, outputs, model_name = LocalizationPipeline(
            model_name=model,
        ).run(
            segments_path=Path(segments_output),
            run_directory=run,
            source_language=manifest.source_language,
            target_language=manifest.target_language,
            glossary_path=glossary,
            context_path=context,
            reuse_completed_batches=not force,
        )
    except LocalizationError as error:
        message = _fail_cli_stage(
            manifest, "localize", measurement, error
        )
        manifest.save(run)
        typer.echo(f"Localize failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    _complete_cli_stage(
        manifest,
        "localize",
        measurement,
        run_status=RunStatus.LOCALIZED,
    )
    stage.outputs = outputs
    manifest.models["translator"] = model_name
    manifest.outputs.update(outputs)
    metrics = TranslationMetrics.model_validate_json(
        Path(outputs["translation_metrics"]).read_text(encoding="utf-8")
    )
    stage.provider = metrics.provider
    stage.model = metrics.model
    stage.input_fingerprint = metrics.configuration_fingerprint
    stage.cost_usd = metrics.cost_usd
    manifest.save(run)

    typer.echo(f"Localize complete: {run}")
    typer.echo(f"Segments: {len(localized_segments)}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def synthesize(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Existing run directory with localized segment outputs.",
    ),
    voice_reference: Path = typer.Option(
        ...,
        "--voice-reference",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "JSON voice catalog (or legacy single reference) with explicit "
            "consent metadata."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Generate a new TTS revision even when outputs exist.",
    ),
    model: str = typer.Option(
        "ai4bharat/IndicF5",
        help="Speech model name to record and load.",
    ),
    require_distinct_voices: bool = typer.Option(
        False,
        "--require-distinct-voices",
        help=(
            "Fail when the voice catalog cannot give every speaker its own "
            "voice, instead of sharing one voice between speakers."
        ),
    ),
) -> None:
    """Generate one Hindi speech audio artifact per localized segment."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("synthesize", StageRecord())
    localized_output = manifest.outputs.get("localized_segments")
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and localized_output is not None
        and synthesis_outputs_reusable(
            outputs=stage.outputs,
            localized_segments_path=Path(localized_output),
            voice_reference_path=voice_reference,
            run_directory=run,
            target_language=manifest.target_language,
            provider_name="indicf5",
            model_name=model,
        )
    ):
        typer.echo(f"Synthesize already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    if not localized_output:
        typer.echo("Synthesize requires localized segments.", err=True)
        raise typer.Exit(code=1)

    measurement = _start_cli_stage(manifest, "synthesize")
    manifest.save(run)

    try:
        synthesized_segments, outputs, model_name = SynthesisPipeline(
            model_name=model,
            require_distinct_voices=require_distinct_voices,
        ).run(
            localized_segments_path=Path(localized_output),
            run_directory=run,
            target_language=manifest.target_language,
            voice_reference_path=voice_reference,
            reuse_completed_utterances=not force,
        )
    except SynthesisError as error:
        message = _fail_cli_stage(
            manifest, "synthesize", measurement, error
        )
        manifest.save(run)
        typer.echo(f"Synthesize failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    _complete_cli_stage(
        manifest,
        "synthesize",
        measurement,
        run_status=RunStatus.SYNTHESIZED,
    )
    stage.outputs = outputs
    manifest.models["tts"] = model_name
    manifest.outputs.update(outputs)
    metrics = SynthesisMetrics.model_validate_json(
        Path(outputs["synthesis_metrics"]).read_text(encoding="utf-8")
    )
    stage.provider = metrics.provider
    stage.model = metrics.model
    stage.input_fingerprint = metrics.configuration_fingerprint
    manifest.save(run)

    typer.echo(f"Synthesize complete: {run}")
    typer.echo(f"Segments: {len(synthesized_segments)}")
    typer.echo(
        "Timing: "
        f"{metrics.duration_within_primary_count}/{metrics.utterance_count} "
        "within primary tolerance; "
        f"{metrics.duration_unresolved_count} unresolved"
    )
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def render(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Existing run directory with synthesized segment outputs.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-render final media even when completed outputs exist.",
    ),
    composition: CompositionMode = typer.Option(
        CompositionMode.CLEAN_REPLACEMENT,
        "--composition",
        help=(
            "Audio background mode: clean replacement, or duck the original "
            "track during dubbed utterances."
        ),
    ),
) -> None:
    """Assemble synthesized speech into subtitle, audio, and video outputs."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("render", StageRecord())
    synthesized_output = manifest.outputs.get("synthesized_segments")
    source_segment = manifest.outputs.get("source_segment")
    policy = RenderPolicy(composition_mode=composition)
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and synthesized_output is not None
        and source_segment is not None
        and (
            render_outputs_reusable(
                outputs=stage.outputs,
                synthesized_segments_path=Path(synthesized_output),
                source_segment_path=Path(source_segment),
                run_directory=run,
                duration_ms=manifest.duration_ms,
                policy=policy,
            )
            or (
                composition == CompositionMode.CLEAN_REPLACEMENT
                and "render_report" not in stage.outputs
                and _render_outputs_exist(stage.outputs)
            )
        )
    ):
        typer.echo(f"Render already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    if not synthesized_output:
        typer.echo("Render requires synthesized segments.", err=True)
        raise typer.Exit(code=1)
    if not source_segment:
        typer.echo("Render requires the ingested source segment.", err=True)
        raise typer.Exit(code=1)

    measurement = _start_cli_stage(manifest, "render")
    manifest.save(run)

    try:
        pipeline = (
            RenderPipeline()
            if composition == CompositionMode.CLEAN_REPLACEMENT
            else RenderPipeline(policy=policy)
        )
        plan, outputs = pipeline.run(
            synthesized_segments_path=Path(synthesized_output),
            source_segment_path=Path(source_segment),
            run_directory=run,
            duration_ms=manifest.duration_ms,
            reuse_completed=not force,
        )
    except RenderError as error:
        message = _fail_cli_stage(manifest, "render", measurement, error)
        manifest.save(run)
        typer.echo(f"Render failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    _complete_cli_stage(
        manifest,
        "render",
        measurement,
        run_status=RunStatus.RENDERED,
    )
    stage.outputs = outputs
    stage.provider = "ffmpeg"
    stage.input_fingerprint = RenderReport.model_validate_json(
        Path(outputs["render_report"]).read_text(encoding="utf-8")
    ).configuration_fingerprint
    manifest.outputs.update(outputs)
    manifest.save(run)

    review_count = sum(1 for segment in plan.segments if segment.needs_review)
    typer.echo(f"Render complete: {run}")
    typer.echo(f"Segments: {len(plan.segments)}")
    typer.echo(f"Needs review: {review_count}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def preflight(
    profile: PreflightProfile = typer.Option(
        PreflightProfile.LOCAL,
        "--profile",
        help=(
            "Readiness profile. 'local' keeps provider checks advisory; "
            "'benchmark' requires the complete GPU/provider/cost/input setup."
        ),
    ),
    run: Path | None = typer.Option(
        None,
        "--run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Optional run directory to inspect for stage artifacts.",
    ),
    voice_reference: Path | None = typer.Option(
        None,
        "--voice-reference",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional voice catalog or legacy reference JSON to validate.",
    ),
    input_video: Path | None = typer.Option(
        None,
        "--input-video",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Authorized 30-45 minute source required by benchmark profile.",
    ),
    target_language: str = typer.Option(
        "hi",
        "--target-language",
        help=(
            "Dubbing target language. Reference transcripts must be written "
            "in this language's script for IndicF5 to time speech correctly."
        ),
    ),
) -> None:
    """Check local or strict long-form benchmark readiness."""
    report = build_preflight_report(
        profile=profile,
        run_directory=run,
        voice_reference_path=voice_reference,
        input_video_path=input_video,
        target_language=target_language,
    )
    typer.echo(report_to_json(report).rstrip())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def ui(
    runs: Path = typer.Option(
        Path("runs"),
        "--runs",
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing saved dubbing runs.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        help="Host interface for the review UI.",
    ),
    port: int = typer.Option(
        8765,
        help="Port for the review UI.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the UI in the default browser.",
    ),
) -> None:
    """Start the customer-facing dubbing review UI."""
    server = UiServer(runs_directory=runs, host=host, port=port)
    typer.echo(f"Dub MVP Studio: {server.url}")
    try:
        server.serve_forever(open_browser=open_browser)
    except KeyboardInterrupt:
        typer.echo("Stopped Dub MVP Studio.")


@app.command()
def web(
    runs: Path = typer.Option(
        Path("runs"),
        "--runs",
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory for customer-created translation jobs.",
    ),
    site: Path = typer.Option(
        Path("site"),
        "--site",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Static customer website directory.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        help="Host interface for the customer web app.",
    ),
    port: int = typer.Option(
        8787,
        help="Port for the customer web app.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the web app in the default browser.",
    ),
    runner: str | None = typer.Option(
        None,
        "--runner",
        help=(
            "Job runner mode: local, queued, or remote. Defaults to "
            "VIDEO_TRANSLATOR_RUNNER or local."
        ),
    ),
) -> None:
    """Start the no-signup customer video translation web app."""
    server = ProductWebServer(
        runs_directory=runs,
        site_directory=site,
        host=host,
        port=port,
        runner_mode=runner,
    )
    typer.echo(f"Video Translator: {server.url}")
    try:
        server.serve_forever(open_browser=open_browser)
    except KeyboardInterrupt:
        typer.echo("Stopped Video Translator.")


@app.command()
def worker(
    runs: Path = typer.Option(
        Path("runs"),
        "--runs",
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing queued translation runs.",
    ),
    once: bool = typer.Option(
        False,
        "--once",
        help="Process at most one queued stage and exit.",
    ),
    poll_seconds: float = typer.Option(
        5.0,
        "--poll-seconds",
        min=0.1,
        help="Seconds between queue scans in daemon mode.",
    ),
) -> None:
    """Process queued stages for the GPU worker runtime."""
    try:
        if once:
            result = run_worker_once(runs_directory=runs)
            if not result.processed:
                typer.echo("No queued jobs.")
                return
            typer.echo(
                f"Processed {result.run_id} {result.stage}: {result.status}"
            )
            return
        typer.echo(f"Worker polling {runs} every {poll_seconds}s")
        run_worker_loop(
            runs_directory=runs,
            poll_seconds=poll_seconds,
        )
    except WorkerError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    except KeyboardInterrupt:
        typer.echo("Stopped worker.")


@app.command()
def benchmark(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Completed or partial run to measure without rerunning stages.",
    ),
    human_review: Path | None = typer.Option(
        None,
        "--human-review",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Completed human-review JSON based on the generated template.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Write a new benchmark revision even when verified reports exist.",
    ),
) -> None:
    """Aggregate durable automated and human quality evidence for a run."""
    try:
        report, artifacts = build_benchmark(
            run,
            human_review_path=human_review,
            reuse_completed=not force,
        )
    except (BenchmarkError, OSError, ValueError) as error:
        typer.echo(f"Benchmark failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    outputs = artifacts.as_outputs(run)
    typer.echo(f"Benchmark complete: {run}")
    typer.echo(f"Release gate: {report.release_gate_status.value}")
    typer.echo(f"JSON: {outputs['benchmark_json']}")
    typer.echo(f"Markdown: {outputs['benchmark_markdown']}")
    if report.human_review["status"] != "completed":
        typer.echo(
            "Human review template: " + outputs["human_review_template"]
        )


@app.command("retry")
def retry_command(
    run: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Run directory containing manifest.json.",
    ),
    from_stage: RetryStage = typer.Option(
        ...,
        "--from",
        help="Earliest stage to invalidate and queue.",
    ),
    utterances: str | None = typer.Option(
        None,
        "--utterances",
        help="Comma-separated stable IDs or numeric suffixes; omit for all.",
    ),
) -> None:
    """Precisely invalidate failed work and its downstream artifacts."""
    selectors = (
        [item.strip() for item in utterances.split(",") if item.strip()]
        if utterances is not None
        else []
    )
    try:
        report = retry_run(
            run,
            from_stage=from_stage,
            utterance_selectors=selectors,
        )
    except (RetryError, OSError, ValueError) as error:
        typer.echo(f"Retry failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Queued {report.queued_stage} for {len(report.affected_utterance_ids)} "
        f"affected utterance(s)."
    )
    typer.echo(f"Retry request: {report.request_id}")
    typer.echo(f"Invalidated sidecars: {len(report.invalidated_sidecars)}")


@app.command("release-check")
def release_check(
    run: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    target: DeploymentTarget = typer.Option(
        DeploymentTarget.LOCAL,
        "--target",
        help="Readiness target; AWS remains blocked until measured prerequisites exist.",
    ),
) -> None:
    """Check benchmark and deployment evidence without changing a run."""
    try:
        report = assess_deployment_readiness(run, target=target)
    except (OSError, ValueError) as error:
        typer.echo(f"Release check failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(readiness_json(report), nl=False)
    if report.status != ReadinessStatus.PASSED:
        raise typer.Exit(code=1)


@app.command("language-check")
def language_check(
    run: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Passing Hindi baseline run.",
    ),
    candidate: str = typer.Option(..., "--candidate"),
    evaluation_set: Path | None = typer.Option(
        None,
        "--evaluation-set",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Enforce Hindi quality and evaluation-set gates before expansion."""
    try:
        report = assess_language_expansion(
            run,
            candidate_language=candidate,
            evaluation_set_path=evaluation_set,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Language check failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(readiness_json(report), nl=False)
    if report.status != ReadinessStatus.PASSED:
        raise typer.Exit(code=1)


@app.command("research-check")
def research_check(
    decision: Path = typer.Option(
        ...,
        "--decision",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Require the X/Y/Z/W/E evidence contract before model training."""
    report = assess_research_readiness(decision)
    typer.echo(readiness_json(report), nl=False)
    if report.status != ReadinessStatus.PASSED:
        raise typer.Exit(code=1)


@app.command()
def status(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Print the same durable run status document used by the web API."""
    try:
        payload = build_run_status(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run status: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(payload.model_dump(mode="json"), indent=2))


@app.command()
def show(
    run: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Print run status (legacy alias for `dub-mvp status`)."""
    try:
        payload = build_run_status(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run status: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(payload.model_dump(mode="json"), indent=2))


def _new_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{timestamp}-{slug or 'run'}"


def _transcribe_outputs_exist(outputs: dict[str, str]) -> bool:
    required = {"whisperx_raw", "transcript", "segments"}
    return required.issubset(outputs) and all(
        Path(outputs[name]).is_file() for name in required
    )


def _segment_outputs_exist(outputs: dict[str, str]) -> bool:
    required = {
        "dubbing_utterances",
        "translation_segments",
        "dubbing_utterances_metadata",
    }
    return required.issubset(outputs) and all(
        Path(outputs[name]).is_file() for name in required
    )


def _render_outputs_exist(outputs: dict[str, str]) -> bool:
    required = {
        "alignment_plan",
        "hindi_srt",
        "dubbed_audio",
        "dubbed_video",
    }
    return required.issubset(outputs) and all(
        Path(outputs[name]).is_file() for name in required
    )

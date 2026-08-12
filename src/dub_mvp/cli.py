from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import typer

from dub_mvp.manifest import (
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
)
from dub_mvp.localize import (
    LocalizationError,
    LocalizationPipeline,
    localization_outputs_reusable,
)
from dub_mvp.media import MediaIngestor, MediaToolError, media_duration_ms
from dub_mvp.preflight import build_preflight_report, report_to_json
from dub_mvp.render import RenderError, RenderPipeline
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

    run_id = _new_run_id(name or input.stem)
    run_directory = output.expanduser().resolve() / run_id
    manifest = RunManifest(
        run_id=run_id,
        source_path=str(input),
        source_start_ms=start_ms,
        source_end_ms=end_ms,
    )
    manifest.save(run_directory)

    stage = manifest.stages["ingest"]
    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    manifest.status = RunStatus.RUNNING
    manifest.save(run_directory)

    started = time.monotonic()
    try:
        metadata, outputs = ingestor.ingest(
            source=input,
            run_directory=run_directory,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    except MediaToolError as error:
        message = str(error)
        stage.status = StageStatus.FAILED
        stage.error = message
        stage.completed_at = datetime.now(timezone.utc)
        manifest.status = RunStatus.FAILED
        manifest.errors.append(message)
        manifest.timings_seconds["ingest"] = time.monotonic() - started
        manifest.save(run_directory)
        typer.echo(f"Ingest failed: {message}", err=True)
        typer.echo(f"Run manifest: {run_directory / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    stage.status = StageStatus.COMPLETED
    stage.completed_at = datetime.now(timezone.utc)
    stage.outputs = outputs
    manifest.status = RunStatus.INGESTED
    manifest.media = metadata
    manifest.outputs.update(outputs)
    manifest.timings_seconds["ingest"] = time.monotonic() - started
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
    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    stage.completed_at = None
    stage.error = None
    manifest.status = RunStatus.RUNNING
    manifest.save(run)

    started = time.monotonic()
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
        message = str(error)
        stage.status = StageStatus.FAILED
        stage.error = message
        stage.completed_at = datetime.now(timezone.utc)
        manifest.status = RunStatus.FAILED
        manifest.errors.append(message)
        manifest.timings_seconds["transcribe"] = time.monotonic() - started
        manifest.save(run)
        typer.echo(f"Transcribe failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    stage.status = StageStatus.COMPLETED
    stage.completed_at = datetime.now(timezone.utc)
    stage.outputs = outputs
    manifest.status = RunStatus.TRANSCRIBED
    manifest.models["whisperx"] = transcript.model
    manifest.outputs.update(outputs)
    manifest.timings_seconds["transcribe"] = time.monotonic() - started
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

    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    stage.completed_at = None
    stage.error = None
    manifest.status = RunStatus.RUNNING
    manifest.save(run)

    started = time.monotonic()
    try:
        artifact, translation_segments, outputs = UtterancePipeline().run(
            transcript_path=Path(transcript_output),
            segments_path=Path(segments_output),
            run_directory=run,
        )
    except UtteranceError as error:
        message = str(error)
        stage.status = StageStatus.FAILED
        stage.error = message
        stage.completed_at = datetime.now(timezone.utc)
        manifest.status = RunStatus.FAILED
        manifest.errors.append(message)
        manifest.timings_seconds["segment"] = time.monotonic() - started
        manifest.save(run)
        typer.echo(f"Segment failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    stage.status = StageStatus.COMPLETED
    stage.completed_at = datetime.now(timezone.utc)
    stage.outputs = outputs
    manifest.status = RunStatus.SEGMENTED
    manifest.outputs.update(outputs)
    manifest.timings_seconds["segment"] = time.monotonic() - started
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

    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    stage.completed_at = None
    stage.error = None
    manifest.status = RunStatus.RUNNING
    manifest.save(run)

    started = time.monotonic()
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
        message = str(error)
        stage.status = StageStatus.FAILED
        stage.error = message
        stage.completed_at = datetime.now(timezone.utc)
        manifest.status = RunStatus.FAILED
        manifest.errors.append(message)
        manifest.timings_seconds["localize"] = time.monotonic() - started
        manifest.save(run)
        typer.echo(f"Localize failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    stage.status = StageStatus.COMPLETED
    stage.completed_at = datetime.now(timezone.utc)
    stage.outputs = outputs
    manifest.status = RunStatus.LOCALIZED
    manifest.models["translator"] = model_name
    manifest.outputs.update(outputs)
    manifest.timings_seconds["localize"] = time.monotonic() - started
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

    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    stage.completed_at = None
    stage.error = None
    manifest.status = RunStatus.RUNNING
    manifest.save(run)

    started = time.monotonic()
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
        message = str(error)
        stage.status = StageStatus.FAILED
        stage.error = message
        stage.completed_at = datetime.now(timezone.utc)
        manifest.status = RunStatus.FAILED
        manifest.errors.append(message)
        manifest.timings_seconds["synthesize"] = time.monotonic() - started
        manifest.save(run)
        typer.echo(f"Synthesize failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    stage.status = StageStatus.COMPLETED
    stage.completed_at = datetime.now(timezone.utc)
    stage.outputs = outputs
    manifest.status = RunStatus.SYNTHESIZED
    manifest.models["tts"] = model_name
    manifest.outputs.update(outputs)
    metrics = SynthesisMetrics.model_validate_json(
        Path(outputs["synthesis_metrics"]).read_text(encoding="utf-8")
    )
    stage.provider = metrics.provider
    stage.input_fingerprint = metrics.configuration_fingerprint
    manifest.timings_seconds["synthesize"] = time.monotonic() - started
    manifest.save(run)

    typer.echo(f"Synthesize complete: {run}")
    typer.echo(f"Segments: {len(synthesized_segments)}")
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
) -> None:
    """Assemble synthesized speech into subtitle, audio, and video outputs."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("render", StageRecord())
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and _render_outputs_exist(stage.outputs)
    ):
        typer.echo(f"Render already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    synthesized_output = manifest.outputs.get("synthesized_segments")
    source_segment = manifest.outputs.get("source_segment")
    if not synthesized_output:
        typer.echo("Render requires synthesized segments.", err=True)
        raise typer.Exit(code=1)
    if not source_segment:
        typer.echo("Render requires the ingested source segment.", err=True)
        raise typer.Exit(code=1)

    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    stage.completed_at = None
    stage.error = None
    manifest.status = RunStatus.RUNNING
    manifest.save(run)

    started = time.monotonic()
    try:
        plan, outputs = RenderPipeline().run(
            synthesized_segments_path=Path(synthesized_output),
            source_segment_path=Path(source_segment),
            run_directory=run,
            duration_ms=manifest.duration_ms,
        )
    except RenderError as error:
        message = str(error)
        stage.status = StageStatus.FAILED
        stage.error = message
        stage.completed_at = datetime.now(timezone.utc)
        manifest.status = RunStatus.FAILED
        manifest.errors.append(message)
        manifest.timings_seconds["render"] = time.monotonic() - started
        manifest.save(run)
        typer.echo(f"Render failed: {message}", err=True)
        typer.echo(f"Run manifest: {run / 'manifest.json'}")
        raise typer.Exit(code=1) from error

    stage.status = StageStatus.COMPLETED
    stage.completed_at = datetime.now(timezone.utc)
    stage.outputs = outputs
    manifest.status = RunStatus.RENDERED
    manifest.outputs.update(outputs)
    manifest.timings_seconds["render"] = time.monotonic() - started
    manifest.save(run)

    review_count = sum(1 for segment in plan.segments if segment.needs_review)
    typer.echo(f"Render complete: {run}")
    typer.echo(f"Segments: {len(plan.segments)}")
    typer.echo(f"Needs review: {review_count}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


@app.command()
def preflight(
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
) -> None:
    """Check local readiness before provisioning or using GPU runtime."""
    report = build_preflight_report(
        run_directory=run,
        voice_reference_path=voice_reference,
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
    """Print the public summary of an existing run."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


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

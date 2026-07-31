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
from dub_mvp.localize import LocalizationError, LocalizationPipeline
from dub_mvp.media import MediaIngestor, MediaToolError
from dub_mvp.synthesize import SynthesisError, SynthesisPipeline
from dub_mvp.timecode import parse_timecode_ms
from dub_mvp.transcribe import TranscriptionError, TranscriptionPipeline

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
    end: str = typer.Option(..., help="End as seconds or HH:MM:SS."),
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
        end_ms = parse_timecode_ms(end)
    except ValueError as error:
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
        metadata, outputs = MediaIngestor().ingest(
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
        and _localize_outputs_exist(stage.outputs)
    ):
        typer.echo(f"Localize already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    segments_output = manifest.outputs.get("segments")
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
        help="JSON voice reference with explicit consent metadata.",
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
) -> None:
    """Generate one Hindi speech audio artifact per localized segment."""
    try:
        manifest = RunManifest.load(run)
    except (OSError, ValueError) as error:
        typer.echo(f"Unable to read run manifest: {error}", err=True)
        raise typer.Exit(code=1) from error

    stage = manifest.stages.setdefault("synthesize", StageRecord())
    if (
        not force
        and stage.status == StageStatus.COMPLETED
        and _synthesize_outputs_exist(stage.outputs)
    ):
        typer.echo(f"Synthesize already complete: {run}")
        typer.echo(json.dumps(manifest.public_summary(), indent=2))
        return

    localized_output = manifest.outputs.get("localized_segments")
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
        ).run(
            localized_segments_path=Path(localized_output),
            run_directory=run,
            target_language=manifest.target_language,
            voice_reference_path=voice_reference,
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
    manifest.timings_seconds["synthesize"] = time.monotonic() - started
    manifest.save(run)

    typer.echo(f"Synthesize complete: {run}")
    typer.echo(f"Segments: {len(synthesized_segments)}")
    typer.echo(json.dumps(manifest.public_summary(), indent=2))


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


def _localize_outputs_exist(outputs: dict[str, str]) -> bool:
    required = {"localization_raw", "localized_segments"}
    return required.issubset(outputs) and all(
        Path(outputs[name]).is_file() for name in required
    )


def _synthesize_outputs_exist(outputs: dict[str, str]) -> bool:
    required = {"synthesis_raw", "synthesized_segments"}
    return required.issubset(outputs) and all(
        Path(outputs[name]).is_file() for name in required
    )

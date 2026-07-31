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
    StageStatus,
)
from dub_mvp.media import MediaIngestor, MediaToolError
from dub_mvp.timecode import parse_timecode_ms

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

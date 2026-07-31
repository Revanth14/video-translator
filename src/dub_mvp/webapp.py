from __future__ import annotations

import json
import mimetypes
import os
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dub_mvp.manifest import RunManifest
from dub_mvp.media import MediaIngestor
from dub_mvp.runner import JobRunner, LocalJobRunner, QueuedJobRunner, StageRequest
from dub_mvp.timecode import parse_timecode_ms
from dub_mvp.ui import UiError, build_customer_run_payload


class WebAppError(RuntimeError):
    pass


@dataclass
class UploadedPart:
    name: str
    filename: str | None
    content: bytes


class WebJobService:
    def __init__(
        self,
        *,
        runs_directory: Path,
        ingestor: MediaIngestor | None = None,
        transcription_pipeline: Any | None = None,
        localization_pipeline: Any | None = None,
        synthesis_pipeline: Any | None = None,
        render_pipeline: Any | None = None,
        runner: JobRunner | None = None,
        start_background_jobs: bool = True,
    ) -> None:
        self.runs_directory = runs_directory.expanduser().resolve()
        self.runner = runner or LocalJobRunner(
            ingestor=ingestor or MediaIngestor(),
            transcription_pipeline=transcription_pipeline,
            localization_pipeline=localization_pipeline,
            synthesis_pipeline=synthesis_pipeline,
            render_pipeline=render_pipeline,
            background=start_background_jobs,
        )

    def create_job(
        self,
        *,
        filename: str,
        content: bytes,
        target_language: str,
        start: str = "0",
        end: str = "90",
    ) -> dict[str, Any]:
        if not content:
            raise WebAppError("Uploaded video is empty.")
        start_ms = _parse_time(start, "start")
        end_ms = _parse_time(end, "end")
        if end_ms <= start_ms:
            raise WebAppError("End time must be greater than start time.")

        run_id = _new_web_run_id(filename)
        run_directory = self.runs_directory / run_id
        input_directory = run_directory / "input"
        input_directory.mkdir(parents=True, exist_ok=True)
        source_path = input_directory / _safe_filename(filename)
        source_path.write_bytes(content)

        manifest = RunManifest(
            run_id=run_id,
            source_path=str(source_path),
            source_start_ms=start_ms,
            source_end_ms=end_ms,
            target_language=target_language,
        )
        manifest.save(run_directory)

        self.runner.submit_ingest(run_directory)
        return build_customer_run_payload(run_directory)

    def run_ingest_now(self, run_directory: Path) -> None:
        self.runner.submit_ingest(run_directory)

    def run_stage(
        self,
        *,
        run_id: str,
        stage: str,
        glossary_content: bytes | None = None,
        voice_reference_content: bytes | None = None,
    ) -> dict[str, Any]:
        run_directory = _safe_join(self.runs_directory, run_id)
        if not run_directory.is_dir():
            raise WebAppError(f"Unknown job: {run_id}")
        if stage not in {"transcribe", "localize", "synthesize", "render"}:
            raise WebAppError(f"Unknown stage: {stage}")

        glossary_path = None
        if glossary_content:
            glossary_path = run_directory / "input" / "glossary.json"
            glossary_path.parent.mkdir(parents=True, exist_ok=True)
            glossary_path.write_bytes(glossary_content)

        voice_reference_path = None
        if stage == "synthesize":
            voice_reference_path = run_directory / "input" / "voice-reference.json"
            voice_reference_path.parent.mkdir(parents=True, exist_ok=True)
            if voice_reference_content:
                voice_reference_path.write_bytes(voice_reference_content)
            elif not voice_reference_path.exists():
                voice_reference_path.write_text(
                    json.dumps(
                        {
                            "reference_id": "generic-web-voice",
                            "path": None,
                            "consent": "generic voice selected in web app",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        self.runner.submit_stage(
            StageRequest(
                run_directory=run_directory,
                stage=stage,
                glossary_path=glossary_path,
                voice_reference_path=voice_reference_path,
            )
        )
        return build_customer_run_payload(run_directory)


class ProductWebServer:
    def __init__(
        self,
        *,
        runs_directory: Path,
        site_directory: Path,
        host: str = "127.0.0.1",
        port: int = 8787,
        runner_mode: str | None = None,
    ) -> None:
        self.runs_directory = runs_directory.expanduser().resolve()
        self.site_directory = site_directory.expanduser().resolve()
        self.host = host
        self.port = port
        self.runner_mode = runner_mode

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self, *, open_browser: bool = True) -> None:
        service = WebJobService(
            runs_directory=self.runs_directory,
            runner=_build_runner(self.runner_mode),
        )
        handler = _handler_factory(
            site_directory=self.site_directory,
            runs_directory=self.runs_directory,
            job_service=service,
        )
        server = ThreadingHTTPServer((self.host, self.port), handler)
        if open_browser:
            threading.Timer(0.3, webbrowser.open, args=(self.url,)).start()
        server.serve_forever()


def _handler_factory(
    *,
    site_directory: Path,
    runs_directory: Path,
    job_service: WebJobService,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_file(site_directory / "index.html")
                elif parsed.path in {"/styles.css", "/app.js"}:
                    self._send_file(site_directory / parsed.path.lstrip("/"))
                elif parsed.path.startswith("/api/jobs/"):
                    run_id = unquote(parsed.path.removeprefix("/api/jobs/"))
                    self._send_json(_load_job(runs_directory, run_id))
                elif parsed.path.startswith("/media/"):
                    self._send_media(runs_directory, parsed.path)
                else:
                    self.send_error(404)
            except (UiError, WebAppError) as error:
                self._send_json({"error": str(error)}, status=400)
            except (OSError, ValueError) as error:
                self._send_json({"error": str(error)}, status=500)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/jobs":
                    self._create_job()
                elif parsed.path.startswith("/api/jobs/") and "/stages/" in parsed.path:
                    self._run_stage(parsed.path)
                else:
                    self.send_error(404)
            except WebAppError as error:
                self._send_json({"error": str(error)}, status=400)
            except (OSError, ValueError) as error:
                self._send_json({"error": str(error)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _create_job(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            parts = _parse_multipart(
                body,
                self.headers.get("Content-Type", ""),
            )
            video = parts.get("video")
            if video is None or video.filename is None:
                raise WebAppError("Upload must include a video file.")
            payload = job_service.create_job(
                filename=video.filename,
                content=video.content,
                target_language=_field(parts, "language", "hi"),
                start=_field(parts, "start", "0"),
                end=_field(parts, "end", "90"),
            )
            self._send_json(payload, status=201)

        def _run_stage(self, path: str) -> None:
            prefix = "/api/jobs/"
            rest = path.removeprefix(prefix)
            run_id, _, stage = rest.partition("/stages/")
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            parts = {}
            content_type = self.headers.get("Content-Type", "")
            if body and "multipart/form-data" in content_type:
                parts = _parse_multipart(body, content_type)
            payload = job_service.run_stage(
                run_id=unquote(run_id),
                stage=unquote(stage),
                glossary_content=(
                    parts["glossary"].content if "glossary" in parts else None
                ),
                voice_reference_content=(
                    parts["voice_reference"].content
                    if "voice_reference" in parts
                    else None
                ),
            )
            self._send_json(payload, status=202)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            content_type = (
                mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, root: Path, request_path: str) -> None:
            relative = unquote(request_path.removeprefix("/media/"))
            path = _safe_join(root, relative)
            self._send_file(path)

        def _send_json(self, payload: Any, *, status: int = 200) -> None:
            body = (
                json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _load_job(runs_directory: Path, run_id: str) -> dict[str, Any]:
    if not run_id:
        raise WebAppError("Missing job id.")
    run_directory = _safe_join(runs_directory, run_id)
    if not run_directory.is_dir():
        raise WebAppError(f"Unknown job: {run_id}")
    return build_customer_run_payload(run_directory)


def _parse_multipart(
    body: bytes,
    content_type: str,
) -> dict[str, UploadedPart]:
    marker = "boundary="
    if marker not in content_type:
        raise WebAppError("Expected multipart form upload.")
    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    delimiter = f"--{boundary}".encode("utf-8")
    parts: dict[str, UploadedPart] = {}
    for raw_part in body.split(delimiter):
        raw_part = raw_part.strip()
        if not raw_part or raw_part == b"--":
            continue
        if raw_part.endswith(b"--"):
            raw_part = raw_part[:-2].strip()
        header_blob, _, content = raw_part.partition(b"\r\n\r\n")
        if not header_blob or not content:
            continue
        headers = _part_headers(header_blob)
        disposition = headers.get("content-disposition", "")
        name = _disposition_value(disposition, "name")
        if not name:
            continue
        filename = _disposition_value(disposition, "filename")
        parts[name] = UploadedPart(
            name=name,
            filename=filename,
            content=content.rstrip(b"\r\n"),
        )
    return parts


def _part_headers(header_blob: bytes) -> dict[str, str]:
    headers = {}
    for line in header_blob.decode("utf-8", errors="replace").split("\r\n"):
        name, _, value = line.partition(":")
        if name and value:
            headers[name.lower()] = value.strip()
    return headers


def _disposition_value(disposition: str, key: str) -> str | None:
    for item in disposition.split(";"):
        name, _, value = item.strip().partition("=")
        if name == key:
            return value.strip().strip('"')
    return None


def _field(
    parts: dict[str, UploadedPart],
    name: str,
    default: str,
) -> str:
    part = parts.get(name)
    if part is None:
        return default
    value = part.content.decode("utf-8", errors="replace").strip()
    return value or default


def _safe_join(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise WebAppError("Path is outside the runs directory.")
    return path


def _safe_filename(filename: str) -> str:
    cleaned = Path(filename).name.strip().replace(" ", "-")
    allowed = [
        character
        for character in cleaned
        if character.isalnum() or character in {"-", "_", "."}
    ]
    return "".join(allowed) or "upload.mp4"


def _new_web_run_id(filename: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(filename).stem.lower()
    slug = "".join(
        character if character.isalnum() else "-"
        for character in stem
    ).strip("-")
    return f"{timestamp}-{slug or 'upload'}-{uuid.uuid4().hex[:8]}"


def _parse_time(value: str, label: str) -> int:
    try:
        return parse_timecode_ms(value)
    except ValueError as error:
        raise WebAppError(f"Invalid {label} time: {value}") from error


def _build_runner(mode: str | None) -> JobRunner:
    selected = (mode or os.getenv("VIDEO_TRANSLATOR_RUNNER", "local")).lower()
    if selected == "local":
        return LocalJobRunner(background=True)
    if selected in {"queued", "remote"}:
        return QueuedJobRunner()
    raise WebAppError(f"Unknown runner mode: {mode}")

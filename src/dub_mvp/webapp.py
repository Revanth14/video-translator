from __future__ import annotations

import json
import mimetypes
import os
import shutil
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from dub_mvp.localize import Glossary, TranslationContext
from dub_mvp.manifest import RunManifest
from dub_mvp.media import MediaIngestor, MediaToolError, media_duration_ms
from dub_mvp.runner import JobRunner, LocalJobRunner, QueuedJobRunner, StageRequest
from dub_mvp.timecode import parse_timecode_ms
from dub_mvp.ui import UiError, build_customer_run_payload
from dub_mvp.upload import StreamedPart, UploadError, parse_multipart_stream
from dub_mvp.worker import run_worker_loop, run_worker_once


MAX_UPLOAD_BYTES = int(
    os.getenv("VIDEO_TRANSLATOR_MAX_UPLOAD_BYTES", str(8 * 1024**3))
)


class WebAppError(RuntimeError):
    pass


class WebJobService:
    def __init__(
        self,
        *,
        runs_directory: Path,
        ingestor: MediaIngestor | None = None,
        transcription_pipeline: Any | None = None,
        utterance_pipeline: Any | None = None,
        localization_pipeline: Any | None = None,
        synthesis_pipeline: Any | None = None,
        render_pipeline: Any | None = None,
        runner: JobRunner | None = None,
        start_background_jobs: bool = True,
    ) -> None:
        self.runs_directory = runs_directory.expanduser().resolve()
        self.media_ingestor = ingestor or MediaIngestor()
        selected_runner = runner or LocalJobRunner(
            ingestor=self.media_ingestor,
            transcription_pipeline=transcription_pipeline,
            utterance_pipeline=utterance_pipeline,
            localization_pipeline=localization_pipeline,
            synthesis_pipeline=synthesis_pipeline,
            render_pipeline=render_pipeline,
            background=False,
        )
        self.runner = QueuedJobRunner()
        self.worker_runner: LocalJobRunner | None = None
        self.start_background_jobs = start_background_jobs
        self._worker_thread: threading.Thread | None = None
        if isinstance(selected_runner, LocalJobRunner):
            selected_runner.background = False
            self.worker_runner = selected_runner
            if start_background_jobs:
                self._worker_thread = threading.Thread(
                    target=run_worker_loop,
                    kwargs={
                        "runs_directory": self.runs_directory,
                        "poll_seconds": 0.25,
                        "runner": self.worker_runner,
                    },
                    daemon=True,
                    name="video-translator-worker",
                )
                self._worker_thread.start()
        elif not isinstance(selected_runner, QueuedJobRunner):
            raise WebAppError(
                "Web jobs require a LocalJobRunner or QueuedJobRunner."
            )

    def create_job(
        self,
        *,
        filename: str,
        source_file: Path,
        target_language: str,
        start: str = "0",
        end: str | None = None,
        glossary_content: bytes | None = None,
        translation_context_content: bytes | None = None,
        voice_reference_content: bytes | None = None,
    ) -> dict[str, Any]:
        """Create a run from an already-uploaded file.

        The upload arrives on disk rather than in memory: a creator video is
        far too large to hold in the web process, let alone on a worker that
        also holds model weights.
        """
        if not source_file.is_file() or source_file.stat().st_size == 0:
            raise WebAppError("Uploaded video is empty.")
        start_ms = _parse_time(start, "start")

        run_id = _new_web_run_id(filename)
        run_directory = self.runs_directory / run_id
        input_directory = run_directory / "input"
        input_directory.mkdir(parents=True, exist_ok=True)
        source_path = input_directory / _safe_filename(filename)
        shutil.move(str(source_file), str(source_path))

        # A rejected upload must not leave a half-built run behind for the
        # worker to discover.
        try:
            try:
                source_duration_ms = media_duration_ms(
                    self.media_ingestor.inspect(source_path)
                )
            except MediaToolError as error:
                raise WebAppError(str(error)) from error
            end_ms = (
                _parse_time(end, "end")
                if end is not None and end.strip()
                else source_duration_ms
            )
            if end_ms <= start_ms:
                raise WebAppError("End time must be greater than start time.")
            if end_ms > source_duration_ms + 100:
                raise WebAppError(
                    "End time exceeds the source duration "
                    f"({source_duration_ms / 1000:.3f}s)."
                )
            glossary = _validated_json_input(
                glossary_content,
                model=Glossary,
                default=Glossary(),
                label="glossary",
            )
            translation_context = _validated_json_input(
                translation_context_content,
                model=TranslationContext,
                default=TranslationContext(),
                label="translation context",
            )
            _write_model_json(input_directory / "glossary.json", glossary)
            _write_model_json(
                input_directory / "translation-context.json",
                translation_context,
            )
            voice_reference_path = input_directory / "voice-reference.json"
            if voice_reference_content:
                voice_reference_path.write_bytes(voice_reference_content)
            else:
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

            manifest = RunManifest(
                run_id=run_id,
                source_path=str(source_path),
                source_start_ms=start_ms,
                source_end_ms=end_ms,
                target_language=target_language,
            )
            manifest.save(run_directory)
        except Exception:
            shutil.rmtree(run_directory, ignore_errors=True)
            raise
        self.runner.submit_ingest(run_directory)
        self._drain_local_worker()
        return build_customer_run_payload(run_directory)

    def run_ingest_now(self, run_directory: Path) -> None:
        self.runner.submit_ingest(run_directory)
        self._drain_local_worker()

    def run_stage(
        self,
        *,
        run_id: str,
        stage: str,
        glossary_content: bytes | None = None,
        translation_context_content: bytes | None = None,
        voice_reference_content: bytes | None = None,
    ) -> dict[str, Any]:
        run_directory = _safe_join(self.runs_directory, run_id)
        if not run_directory.is_dir():
            raise WebAppError(f"Unknown job: {run_id}")
        if stage not in {
            "transcribe",
            "segment",
            "localize",
            "synthesize",
            "render",
        }:
            raise WebAppError(f"Unknown stage: {stage}")

        glossary_path = None
        if glossary_content:
            glossary_path = run_directory / "input" / "glossary.json"
            glossary_path.parent.mkdir(parents=True, exist_ok=True)
            _write_model_json(
                glossary_path,
                _validated_json_input(
                    glossary_content,
                    model=Glossary,
                    default=Glossary(),
                    label="glossary",
                ),
            )

        translation_context_path = None
        if translation_context_content:
            translation_context_path = (
                run_directory / "input" / "translation-context.json"
            )
            _write_model_json(
                translation_context_path,
                _validated_json_input(
                    translation_context_content,
                    model=TranslationContext,
                    default=TranslationContext(),
                    label="translation context",
                ),
            )

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
                translation_context_path=translation_context_path,
                voice_reference_path=voice_reference_path,
            )
        )
        self._drain_local_worker()
        return build_customer_run_payload(run_directory)

    def _drain_local_worker(self) -> None:
        if self.worker_runner is None or self.start_background_jobs:
            return
        while run_worker_once(
            runs_directory=self.runs_directory,
            runner=self.worker_runner,
        ).processed:
            pass


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
            except (WebAppError, UploadError) as error:
                self._send_json({"error": str(error)}, status=400)
            except (OSError, ValueError) as error:
                self._send_json({"error": str(error)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _create_job(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > MAX_UPLOAD_BYTES:
                raise WebAppError(
                    "Upload exceeds the maximum size of "
                    f"{MAX_UPLOAD_BYTES // (1024 ** 3)} GB."
                )
            staging_directory = runs_directory / ".uploads"
            staged: list[Path] = []

            def sink(name: str, filename: str | None) -> Path | None:
                # Only the video streams to disk; the rest are small fields.
                if name != "video" or not filename:
                    return None
                destination = staging_directory / f"{uuid.uuid4().hex}.upload"
                staged.append(destination)
                return destination

            try:
                parts = parse_multipart_stream(
                    self.rfile,
                    content_type=self.headers.get("Content-Type", ""),
                    content_length=content_length,
                    file_sink=sink,
                )
                video = parts.get("video")
                if video is None or video.path is None or not video.filename:
                    raise WebAppError("Upload must include a video file.")
                payload = job_service.create_job(
                    filename=video.filename,
                    source_file=video.path,
                    target_language=_field(parts, "language", "hi"),
                    start=_field(parts, "start", "0"),
                    end=_optional_field(parts, "end"),
                    glossary_content=(
                        parts["glossary"].content
                        if "glossary" in parts
                        else None
                    ),
                    translation_context_content=(
                        parts["translation_context"].content
                        if "translation_context" in parts
                        else None
                    ),
                    voice_reference_content=(
                        parts["voice_reference"].content
                        if "voice_reference" in parts
                        else None
                    ),
                )
            finally:
                # create_job moves the upload into the run on success; anything
                # still staged belongs to a failed request.
                for path in staged:
                    path.unlink(missing_ok=True)
            self._send_json(payload, status=201)

        def _run_stage(self, path: str) -> None:
            prefix = "/api/jobs/"
            rest = path.removeprefix(prefix)
            run_id, _, stage = rest.partition("/stages/")
            content_length = int(self.headers.get("Content-Length", "0"))
            content_type = self.headers.get("Content-Type", "")
            parts: dict[str, StreamedPart] = {}
            if content_length > 0 and "multipart/form-data" in content_type:
                # Operator inputs only (glossary, voice reference), so every
                # part stays buffered under the field limit.
                parts = parse_multipart_stream(
                    self.rfile,
                    content_type=content_type,
                    content_length=content_length,
                )
            payload = job_service.run_stage(
                run_id=unquote(run_id),
                stage=unquote(stage),
                glossary_content=(
                    parts["glossary"].content if "glossary" in parts else None
                ),
                translation_context_content=(
                    parts["translation_context"].content
                    if "translation_context" in parts
                    else None
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





def _field(
    parts: dict[str, StreamedPart],
    name: str,
    default: str,
) -> str:
    part = parts.get(name)
    if part is None:
        return default
    value = part.content.decode("utf-8", errors="replace").strip()
    return value or default


def _optional_field(
    parts: dict[str, StreamedPart],
    name: str,
) -> str | None:
    part = parts.get(name)
    if part is None:
        return None
    return part.content.decode("utf-8", errors="replace").strip() or None


def _validated_json_input(
    content: bytes | None,
    *,
    model: type[Any],
    default: Any,
    label: str,
) -> Any:
    if content is None:
        return default
    try:
        return model.model_validate_json(content)
    except (ValueError, ValidationError) as error:
        raise WebAppError(f"Invalid {label}: {error}") from error


def _write_model_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value.model_dump(mode="json"), indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


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
        return LocalJobRunner(background=False)
    if selected in {"queued", "remote"}:
        return QueuedJobRunner()
    raise WebAppError(f"Unknown runner mode: {mode}")

from __future__ import annotations

import json
import mimetypes
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import BaseModel, Field

from dub_mvp.manifest import RunManifest


class UiError(RuntimeError):
    pass


class UiServer:
    def __init__(
        self,
        *,
        runs_directory: Path,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.runs_directory = runs_directory.expanduser().resolve()
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self, *, open_browser: bool = True) -> None:
        handler = _handler_factory(self.runs_directory)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        if open_browser:
            threading.Timer(0.3, webbrowser.open, args=(self.url,)).start()
        self._server.serve_forever()


class CustomerRunSummary(BaseModel):
    run_id: str
    status: str
    source: str
    range_ms: list[int]
    target_language: str
    updated_at: str
    stages: dict[str, str]
    outputs: dict[str, str]
    metrics: dict[str, Any] = Field(default_factory=dict)


def build_customer_run_payload(run_directory: Path) -> dict[str, Any]:
    manifest = RunManifest.load(run_directory)
    summary = CustomerRunSummary(
        run_id=manifest.run_id,
        status=manifest.status.value,
        source=manifest.source_path,
        range_ms=[manifest.source_start_ms, manifest.source_end_ms],
        target_language=manifest.target_language,
        updated_at=manifest.updated_at.isoformat(),
        stages={
            name: stage.status.value for name, stage in manifest.stages.items()
        },
        outputs=manifest.outputs,
        metrics=_run_metrics(manifest, run_directory),
    )
    return {
        "summary": summary.model_dump(mode="json"),
        "segments": _read_json_output(
            manifest.outputs.get("localized_segments")
            or manifest.outputs.get("translation_segments")
            or manifest.outputs.get("segments")
        ),
        "synthesized_segments": _read_json_output(
            manifest.outputs.get("synthesized_segments")
        ),
        "alignment_plan": _read_json_output(
            manifest.outputs.get("alignment_plan")
        ),
        "errors": manifest.errors,
    }


def list_customer_runs(runs_directory: Path) -> list[dict[str, Any]]:
    if not runs_directory.is_dir():
        return []
    runs = []
    for manifest_path in sorted(
        runs_directory.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        try:
            manifest = RunManifest.load(manifest_path.parent)
        except (OSError, ValueError):
            continue
        runs.append(
            {
                "run_id": manifest.run_id,
                "status": manifest.status.value,
                "updated_at": manifest.updated_at.isoformat(),
                "duration_ms": manifest.duration_ms,
                "target_language": manifest.target_language,
            }
        )
    return runs


def demo_payload() -> dict[str, Any]:
    return {
        "summary": {
            "run_id": "demo-preview",
            "status": "rendered",
            "source": "Founder product walkthrough.mp4",
            "range_ms": [0, 90000],
            "target_language": "hi",
            "updated_at": datetime.now().isoformat(),
            "stages": {
                "ingest": "completed",
                "transcribe": "completed",
                "segment": "completed",
                "localize": "completed",
                "synthesize": "completed",
                "render": "completed",
            },
            "outputs": {},
            "metrics": {
                "segments": 12,
                "localized": 12,
                "synthesized": 12,
                "needs_review": 1,
                "duration": "1:30",
            },
        },
        "segments": [
            {
                "segment_id": "seg_0001",
                "start_ms": 1240,
                "end_ms": 6980,
                "duration_budget_ms": 5740,
                "source_text": "Let me show you how deployment works.",
                "target_text": "Main aapko dikhata hoon ki deployment kaise kaam karta hai.",
            },
            {
                "segment_id": "seg_0002",
                "start_ms": 7460,
                "end_ms": 11880,
                "duration_budget_ms": 4420,
                "source_text": "The API server receives the request.",
                "target_text": "API server request receive karta hai.",
            },
            {
                "segment_id": "seg_0003",
                "start_ms": 12300,
                "end_ms": 18400,
                "duration_budget_ms": 6100,
                "source_text": "Then we build the Docker image.",
                "target_text": "Phir hum Docker image build karte hain.",
            },
        ],
        "synthesized_segments": [
            {
                "segment_id": "seg_0001",
                "tts_duration_ms": 5900,
                "tts_audio_path": "",
            },
            {
                "segment_id": "seg_0002",
                "tts_duration_ms": 4210,
                "tts_audio_path": "",
            },
            {
                "segment_id": "seg_0003",
                "tts_duration_ms": 6650,
                "tts_audio_path": "",
            },
        ],
        "alignment_plan": {
            "segments": [
                {"segment_id": "seg_0001", "needs_review": False},
                {"segment_id": "seg_0002", "needs_review": False},
                {"segment_id": "seg_0003", "needs_review": True},
            ]
        },
        "errors": [],
    }


def _handler_factory(runs_directory: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(HTML)
                elif parsed.path == "/api/runs":
                    self._send_json({"runs": list_customer_runs(runs_directory)})
                elif parsed.path == "/api/demo":
                    self._send_json(demo_payload())
                elif parsed.path == "/api/run":
                    query = parse_qs(parsed.query)
                    run_id = query.get("id", [""])[0]
                    self._send_json(_load_run_payload(runs_directory, run_id))
                elif parsed.path.startswith("/media/"):
                    self._send_media(runs_directory, parsed.path)
                else:
                    self.send_error(404)
            except UiError as error:
                self._send_json({"error": str(error)}, status=400)
            except (OSError, ValueError) as error:
                self._send_json({"error": str(error)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, *, status: int = 200) -> None:
            body = (
                json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, root: Path, request_path: str) -> None:
            relative = unquote(request_path.removeprefix("/media/"))
            path = _safe_join(root, relative)
            if not path.is_file():
                self.send_error(404)
                return
            content_type = (
                mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 256):
                    self.wfile.write(chunk)

    return Handler


def _load_run_payload(runs_directory: Path, run_id: str) -> dict[str, Any]:
    if not run_id:
        raise UiError("Missing run id.")
    run_directory = _safe_join(runs_directory, run_id)
    if not run_directory.is_dir():
        raise UiError(f"Unknown run: {run_id}")
    return build_customer_run_payload(run_directory)


def _safe_join(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise UiError("Path is outside the runs directory.")
    return path


def _run_metrics(
    manifest: RunManifest,
    run_directory: Path,
) -> dict[str, Any]:
    segments = _as_list(
        _read_json_output(
            manifest.outputs.get("translation_segments")
            or manifest.outputs.get("segments")
        )
    )
    localized = _as_list(
        _read_json_output(manifest.outputs.get("localized_segments"))
    )
    synthesized = _as_list(
        _read_json_output(manifest.outputs.get("synthesized_segments"))
    )
    alignment = _read_json_output(manifest.outputs.get("alignment_plan"))
    needs_review = 0
    if isinstance(alignment, dict):
        needs_review = sum(
            1 for item in alignment.get("segments", [])
            if isinstance(item, dict) and item.get("needs_review")
        )
    return {
        "segments": len(segments),
        "localized": len(localized),
        "synthesized": len(synthesized),
        "needs_review": needs_review,
        "duration": _duration_label(manifest.duration_ms),
        "has_video": _relative_media(
            run_directory,
            manifest.outputs.get("dubbed_video"),
        )
        is not None,
    }


def _read_json_output(path: str | None) -> Any:
    if not path:
        return None
    output_path = Path(path)
    if not output_path.is_file():
        return None
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _duration_label(milliseconds: int) -> str:
    total_seconds = max(0, milliseconds // 1000)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _relative_media(run_directory: Path, path: str | None) -> str | None:
    if not path:
        return None
    output_path = Path(path)
    if not output_path.is_file():
        return None
    try:
        return str(output_path.resolve().relative_to(run_directory.resolve()))
    except ValueError:
        return None


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dub MVP Studio</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18201d;
      --muted: #64706b;
      --line: #d9e0dc;
      --panel: #ffffff;
      --wash: #f5f7f2;
      --brand: #0f8b6f;
      --brand-dark: #08624e;
      --accent: #d85f33;
      --ok: #12805c;
      --warn: #b7791f;
      --fail: #b42318;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--wash);
    }
    button, select {
      font: inherit;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
    }
    aside {
      background: #17231f;
      color: #f8fbf8;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }
    .brand {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .brand strong {
      font-size: 20px;
      line-height: 1.1;
    }
    .brand span, .hint {
      color: #b8c5bd;
      font-size: 13px;
      line-height: 1.5;
    }
    .run-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .run-button {
      width: 100%;
      border: 1px solid rgba(255,255,255,.16);
      background: rgba(255,255,255,.06);
      color: #f8fbf8;
      padding: 10px 12px;
      border-radius: 8px;
      text-align: left;
      cursor: pointer;
    }
    .run-button.active {
      border-color: #7ad9c2;
      background: rgba(122,217,194,.18);
    }
    main {
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .topbar, .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .topbar {
      padding: 18px 20px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 26px;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 17px;
      letter-spacing: 0;
    }
    .subtle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .button {
      border: 1px solid var(--brand);
      background: var(--brand);
      color: white;
      border-radius: 8px;
      padding: 9px 12px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
    }
    .button.secondary {
      color: var(--brand-dark);
      background: white;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
      gap: 20px;
    }
    .section {
      padding: 18px;
    }
    .video-frame {
      aspect-ratio: 16 / 9;
      background: #101816;
      border-radius: 8px;
      overflow: hidden;
      display: grid;
      place-items: center;
      color: #dbe7e1;
      position: relative;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #101816;
    }
    .preview-mark {
      width: min(72%, 520px);
      aspect-ratio: 16 / 9;
      border: 1px solid rgba(255,255,255,.2);
      border-radius: 8px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px;
      background:
        linear-gradient(90deg, rgba(255,255,255,.09) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,.09) 1px, transparent 1px);
      background-size: 36px 36px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 72px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
      display: block;
      margin-bottom: 6px;
    }
    .metric strong {
      font-size: 22px;
    }
    .stages {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .stage {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
    }
    .pill {
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      background: #edf2ef;
      color: var(--muted);
    }
    .pill.completed, .pill.rendered, .pill.localized, .pill.synthesized {
      background: #e5f5ee;
      color: var(--ok);
    }
    .pill.failed { background: #fee4df; color: var(--fail); }
    .pill.running { background: #fff4d6; color: var(--warn); }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
      font-size: 14px;
      overflow-wrap: break-word;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .time {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      width: 88px;
    }
    audio {
      width: 100%;
      min-width: 180px;
    }
    .empty {
      padding: 24px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: static; }
      .grid, .metrics { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      main { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <strong>Dub MVP Studio</strong>
        <span>Private Hindi dubbing review</span>
      </div>
      <button class="run-button active" id="demoButton">Demo Preview</button>
      <div>
        <div class="hint">Available Runs</div>
        <div class="run-list" id="runList"></div>
      </div>
      <p class="hint">Show the customer the video, proof of translation, timing health, and downloadable deliverables from one calm screen.</p>
    </aside>
    <main>
      <section class="topbar">
        <div>
          <h1 id="title">Demo Preview</h1>
          <div class="subtle" id="subtitle">Customer-facing dubbing review</div>
        </div>
        <div class="actions" id="downloads"></div>
      </section>

      <section class="grid">
        <div class="section">
          <h2>Dubbed Video</h2>
          <div class="video-frame" id="videoFrame"></div>
        </div>
        <div class="section">
          <h2>Run Progress</h2>
          <div class="stages" id="stages"></div>
        </div>
      </section>

      <section class="section">
        <h2>Delivery Snapshot</h2>
        <div class="metrics" id="metrics"></div>
      </section>

      <section class="section">
        <h2>Segment Review</h2>
        <div id="segments"></div>
      </section>
    </main>
  </div>
  <script>
    const state = { activeRun: "demo" };
    const stageLabels = {
      ingest: "Media prepared",
      transcribe: "English transcript",
      segment: "Dubbing utterances",
      localize: "Hindi adaptation",
      synthesize: "Hindi voice",
      render: "Final video"
    };

    async function getJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function boot() {
      document.getElementById("demoButton").addEventListener("click", () => loadDemo());
      const runs = await getJson("/api/runs");
      renderRuns(runs.runs || []);
      await loadDemo();
    }

    function renderRuns(runs) {
      const list = document.getElementById("runList");
      list.innerHTML = "";
      if (!runs.length) {
        list.innerHTML = "<div class='hint'>No saved runs yet.</div>";
        return;
      }
      runs.forEach(run => {
        const button = document.createElement("button");
        button.className = "run-button";
        button.textContent = `${run.run_id} - ${run.status}`;
        button.addEventListener("click", () => loadRun(run.run_id));
        list.appendChild(button);
      });
    }

    async function loadDemo() {
      state.activeRun = "demo";
      markActive();
      renderPayload(await getJson("/api/demo"), "demo");
    }

    async function loadRun(runId) {
      state.activeRun = runId;
      markActive();
      renderPayload(await getJson(`/api/run?id=${encodeURIComponent(runId)}`), runId);
    }

    function markActive() {
      document.querySelectorAll(".run-button").forEach(button => button.classList.remove("active"));
      if (state.activeRun === "demo") document.getElementById("demoButton").classList.add("active");
    }

    function renderPayload(payload, runId) {
      const summary = payload.summary;
      document.getElementById("title").textContent = summary.run_id;
      document.getElementById("subtitle").textContent = `${summary.status} - ${summary.metrics.duration} - ${summary.target_language.toUpperCase()}`;
      renderVideo(summary, runId);
      renderStages(summary.stages);
      renderMetrics(summary.metrics);
      renderDownloads(summary, runId);
      renderSegments(payload, runId);
    }

    function renderVideo(summary, runId) {
      const frame = document.getElementById("videoFrame");
      const videoPath = summary.outputs.dubbed_video;
      if (runId !== "demo" && videoPath) {
        frame.innerHTML = `<video controls preload="metadata" src="/media/${encodeURIComponent(relativePath(videoPath, runId))}"></video>`;
      } else {
        frame.innerHTML = "<div class='preview-mark'><div><strong>Hindi Dub Preview</strong><br><span class='subtle'>The final rendered video appears here after the render stage.</span></div></div>";
      }
    }

    function renderStages(stages) {
      const box = document.getElementById("stages");
      box.innerHTML = "";
      Object.entries(stageLabels).forEach(([key, label]) => {
        const status = stages[key] || "pending";
        const row = document.createElement("div");
        row.className = "stage";
        row.innerHTML = `<span>${label}</span><span class="pill ${status}">${status}</span>`;
        box.appendChild(row);
      });
    }

    function renderMetrics(metrics) {
      document.getElementById("metrics").innerHTML = [
        ["Segments", metrics.segments || 0],
        ["Localized", metrics.localized || 0],
        ["Voiced", metrics.synthesized || 0],
        ["Review", metrics.needs_review || 0],
      ].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
    }

    function renderDownloads(summary, runId) {
      const downloads = document.getElementById("downloads");
      if (runId === "demo") {
        downloads.innerHTML = "<button class='button secondary'>Demo mode</button>";
        return;
      }
      const links = [
        ["Video", summary.outputs.dubbed_video],
        ["Subtitles", summary.outputs.hindi_srt],
        ["Manifest", `${runId}/manifest.json`],
      ].filter(([, value]) => value);
      downloads.innerHTML = links.map(([label, path]) => {
        const media = path.endsWith("/manifest.json") ? path : relativePath(path, runId);
        return `<a class="button" href="/media/${encodeURIComponent(media)}" target="_blank">${label}</a>`;
      }).join("");
    }

    function renderSegments(payload, runId) {
      const segments = Array.isArray(payload.segments) ? payload.segments : [];
      const synth = new Map((payload.synthesized_segments || []).map(item => [item.segment_id, item]));
      const review = new Set(((payload.alignment_plan || {}).segments || []).filter(item => item.needs_review).map(item => item.segment_id));
      if (!segments.length) {
        document.getElementById("segments").innerHTML = "<div class='empty'>Segments appear after transcription.</div>";
        return;
      }
      const rows = segments.map(segment => {
        const voiced = synth.get(segment.segment_id) || {};
        const audio = runId !== "demo" && voiced.tts_audio_path
          ? `<audio controls preload="none" src="/media/${encodeURIComponent(relativePath(voiced.tts_audio_path, runId))}"></audio>`
          : "<span class='subtle'>Pending voice</span>";
        const badge = review.has(segment.segment_id) ? "<span class='pill running'>review</span>" : "<span class='pill completed'>ok</span>";
        return `<tr>
          <td class="time">${time(segment.start_ms)}<br>${time(segment.end_ms)}</td>
          <td>${escapeHtml(segment.source_text || "")}</td>
          <td>${escapeHtml(segment.target_text || "")}</td>
          <td>${audio}</td>
          <td>${badge}</td>
        </tr>`;
      }).join("");
      document.getElementById("segments").innerHTML = `<table>
        <thead><tr><th>Time</th><th>English</th><th>Hindi</th><th>Voice</th><th>Status</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    }

    function relativePath(path, runId) {
      const marker = `/${runId}/`;
      const index = path.indexOf(marker);
      return index >= 0 ? `${runId}/${path.slice(index + marker.length)}` : path;
    }

    function time(ms) {
      const total = Math.floor((ms || 0) / 1000);
      const minutes = Math.floor(total / 60);
      const seconds = total % 60;
      return `${minutes}:${String(seconds).padStart(2, "0")}`;
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    boot().catch(error => {
      document.body.innerHTML = `<main><section class="section"><h1>Unable to load UI</h1><p>${escapeHtml(error.message)}</p></section></main>`;
    });
  </script>
</body>
</html>
"""

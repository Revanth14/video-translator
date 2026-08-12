# Current Architecture

Describes the system as it stands. **Update this file in the same change that
alters the architecture** — a stale architecture doc is worse than none,
because agents and new contributors trust it.

Last verified: 2026-08-12 (110 tests passing).

## Non-negotiable invariants

1. **The manifest is the authority.** Workers, threads, browser tabs, and web
   processes are disposable. Durable state lives in `manifest.json`.
2. **One executor, many environments.** The same worker loop runs embedded in
   the web process and as a standalone GPU worker. Local and remote differ only
   in *where* the loop runs.
3. **Completed work is never redone.** Reuse requires a verified sidecar, never
   file existence.

## Entry points

`dub-mvp` (Typer CLI): `ingest`, `transcribe`, `segment`, `localize`,
`synthesize`, `render`, `preflight`, `ui`, `web`, `worker`, `show`.

The CLI is an **operator and debugging tool**, not the creator path. Creators
use the web app; the CLI must never become the only way to do something.

`ProductWebServer` (`webapp.py`) serves the customer app from `site/` on
`ThreadingHTTPServer`. `UiServer` (`ui.py`) serves the internal review UI.

## Stage graph

```text
ingest → transcribe → segment → localize → synthesize → render
```

Canonical order lives in `manifest.PIPELINE_STAGE_NAMES` and
`worker.STAGE_ORDER`; these must agree.

- `RunStatus`: created, queued, running, ingested, transcribed, segmented,
  localized, synthesized, rendered, failed, cancelled.
- `StageStatus`: pending, queued, running, completed, failed, cancelled,
  invalidated.

## Execution model

### Durable state and concurrency

`RunManifest.save()` and `mutate_manifest()` write through an exclusive
`flock` on `.manifest.lock`, compare the on-disk `revision` against the
in-memory one, then write via temp file → fsync → `os.replace`. A mismatch
raises `ManifestConflictError`.

`mutate_manifest(run_directory, apply)` is the only correct way to do a
read-modify-write: it loads and writes inside one lock, so the callback never
races. **Never hold a `RunManifest` across slow work and save it afterwards** —
a concurrent heartbeat invalidates the write.

`flock` is advisory and POSIX-local. This is safe for workers sharing a local
disk only; remote deployment needs a state store with real conditional writes.
`state_store.py` defines that seam (`RunStateStore` protocol,
`LocalManifestStateStore` implementation).

### Leases and fencing

A claim grants a `Lease(worker_id, lease_generation)`. `lease_generation` is a
fencing token that increments on every claim, so a worker that stalled and lost
its lease cannot publish over the worker that reclaimed the stage.

- `claim_job` — claims a stage, records the attempt, grants the lease. A stage
  that has exhausted `max_attempts` is moved to terminal `FAILED` here, so it
  stops being claimable and cannot starve the runs behind it.
- `LeaseKeeper` — background thread renewing `heartbeat_at` / `lease_expires_at`
  while a stage runs, so a long transcription is not reclaimed mid-flight.
- `begin_stage` / `complete_stage` / `fail_stage` / `renew_lease` — all verify
  lease ownership before writing and return `None` (or `False`) when fenced out.
- `is_claimable` — queued work becomes eligible after its retry backoff;
  running work becomes eligible again once its lease expires (reclamation).

Retries are bounded: `attempt_count` vs `max_attempts`, with exponential
backoff via `retry_delay_seconds` (30s base, 600s cap). Every attempt is
appended to `StageRecord.attempts`, and transitions to `StageRecord.events`.

### Automatic progression

`queue_next_ready_stage` queues the first pending stage whose predecessor is
complete, re-deriving the candidate under the lock. Progression is therefore a
property of durable state, not something a caller must remember to trigger —
this is what closes the "stage completed but successor never queued" gap and
what lets a creator close the browser.

### The worker loop

`run_worker_once` scans runs (oldest `manifest.json` mtime first), advances and
inspects each via `_scan_run`, claims the first eligible stage, runs it under a
`LeaseKeeper`, then queues the successor. A run that cannot be claimed or
written is added to a per-pass `unavailable` set so it cannot block the runs
behind it; the set resets each pass, so a recovered run is picked up again.

`run_worker_loop` is the supervisor: it catches broadly, logs, backs off
(capped at `MAX_LOOP_BACKOFF_SECONDS`), and continues. It must never exit — in
the web app it is a daemon thread, so its death would silently strand every job
while the server kept answering requests.

### Deployment modes

- **Local** — `WebJobService` always submits through `QueuedJobRunner`, and
  runs `run_worker_loop` in a background thread with a synchronous
  `LocalJobRunner`. Do not start an embedded worker per process if the web
  server ever becomes multi-process; use a standalone worker instead.
- **Queued/remote** — the web app only enqueues; `dub-mvp worker` runs the same
  loop on the GPU host. See `docs/gpu-worker-contract.md`.

`LocalJobRunner._run_stage` marks the stage running, executes, and records the
outcome through short locked mutations. It catches known pipeline errors
(retryable under a lease) and any unexpected exception (terminal,
`unexpected_error`), so a stage never sits in `RUNNING` with no explanation.

## Artifacts

Run layout:

```text
runs/<run_id>/
├── manifest.json
├── input/          source upload, glossary.json, voice-reference.json
├── metadata/       ffprobe, whisperx_raw, transcript, segments,
│                   localization_raw, localized_segments,
│                   synthesized_segments, alignment_plan, job-queue.jsonl
├── utterances/     dubbing_utterances.json, translation_segments.json,
│                   dubbing_utterances.meta.json
├── working/        source_segment.mp4, source_audio.wav, dubbed_audio.wav
├── segments/       <segment>/tts-rN.wav  (revisioned, never overwritten)
├── subtitles/      hi.srt
└── outputs/        dubbed_video.mp4
```

`artifacts.py` defines the reuse contract. An artifact is reusable only when
its sidecar exists, status is completed, the file exists, and size, SHA-256,
and input fingerprint all match. `path` is stored **relative to the run
directory** so a run survives being moved, copied, or uploaded to S3.

`fingerprint_inputs` canonicalizes JSON (sorted keys) and rejects bare
datetimes — a timestamp in a fingerprint means nothing is ever reusable again.

## Media contracts

Three distinct concepts, deliberately not interchangeable:

- **`TranscriptWord` / `TranscriptUtterance`** — normalized WhisperX output.
  Words are ordered and non-overlapping; utterances likewise (enforced by
  validators).
- **`TranscriptSegment`** — provider-oriented timing chunk, `segment_id`.
- **`DubbingUtterance`** — the product contract, `utterance_id`: one speaker, a
  duration budget, traceability to transcript word indexes, and neighbouring
  context. Produced by the `segment` stage (`UtterancePipeline`), consumed by
  localization via `translation_segments`.

`OverlapStatus` exists and the artifact validator enforces that overlapping
utterances are marked explicitly, but **nothing currently produces anything but
`NONE`** — overlap detection is not yet implemented. Overlapping input is
rejected upstream by the transcript validators instead.

Localization validates missing, duplicate, unknown, and empty segment results.
Synthesis writes revisioned audio rather than overwriting.

## Web behaviour

- Uploads **stream to disk** (`upload.py`, `parse_multipart_stream`). The video
  part is written to `runs/.uploads/<uuid>.upload` as it arrives and moved into
  the run; every other field is buffered under `MAX_FIELD_BYTES`. Peak memory is
  ~4 MB regardless of upload size. `MAX_UPLOAD_BYTES` (env
  `VIDEO_TRANSLATOR_MAX_UPLOAD_BYTES`, default 8 GB) rejects oversized uploads
  from `Content-Length` before any body is read.
- A rejected upload removes both the staged file and the partially built run
  directory, so the worker never discovers a half-created run.
- Uploads process the **full source duration** by default; duration is derived
  from `MediaIngestor.inspect()` (ffprobe) at job creation. Trimming is an
  explicit advanced option, validated against the real duration.
- Glossary and voice reference are persisted at job creation, before the
  pipeline reaches the stages that need them.
- The browser **only submits and observes**. It never advances stages.
- Run identity persists in `localStorage` and the `?job=<run_id>` URL, so a
  reload or a later visit restores the run. A permanently missing job (400/404)
  releases the stored id; transient errors keep polling.

## Known gaps

- `cli.py` still holds a manifest across pipeline execution (the pattern fixed
  in `runner.py`). Safe only because nothing else writes during a CLI run; it
  will conflict if an operator runs a stage while a worker holds a lease.
- Overlap detection, duration-aware correction, benchmarking, and structured
  per-run event logs are not implemented yet.
- The web app has no upload progress reporting; a large upload is silent until
  it completes.
- No authentication: a run id is not authorization. This is a controlled demo,
  not a public service.

## Environment

- `pyproject.toml` declares only `pydantic` and `typer`. The real runtime
  (ffmpeg/ffprobe, WhisperX, OpenAI, IndicF5, torch) is undeclared and provided
  by `scripts/bootstrap-gpu.sh`. Containerization (Phase 14) must capture it.
- Python is pinned to `>=3.10,<3.11`.
- Run the suite with `uv run python -m pytest` (plain `uv run pytest` may fail
  to spawn).

# GPU Droplet Runbook

This runbook describes the current queued-worker deployment scaffold. It is an
operator reference, not the final AWS architecture.

## Prerequisites

- An Ubuntu GPU host with a compatible NVIDIA driver
- The repository copied or cloned to `/srv/video-translator`
- A shared `runs/` directory visible to both web and worker processes
- Provider credentials configured outside version control

## Bootstrap

From the repository root:

```bash
sudo PROJECT_DIR=/srv/video-translator scripts/bootstrap-gpu.sh
```

Copy `.env.example` to `.env`, fill required credentials, and load those values
using the deployment environment's secret-management mechanism.

## Start the queued web application

```bash
uv run dub-mvp web \
  --runner queued \
  --runs /srv/video-translator/runs \
  --host 0.0.0.0 \
  --port 8787 \
  --no-open
```

## Start the worker

Run continuously:

```bash
uv run dub-mvp worker \
  --runs /srv/video-translator/runs \
  --poll-seconds 5
```

For a one-shot operator check:

```bash
uv run dub-mvp worker --runs /srv/video-translator/runs --once
```

## Operator Smoke Test

1. Run `uv run dub-mvp preflight` and confirm FFmpeg and FFprobe pass.
2. Start the web application in queued mode.
3. Upload a short authorized video.
4. Run the worker once and verify the queued ingest stage is processed.
5. Inspect the run's `manifest.json` and confirm ingest completed.
6. Start the continuous worker only after the one-shot check succeeds.

## Current limitations

- The current worker contract does not yet provide multi-worker atomic claims,
  heartbeats, fencing tokens, or automatic full-pipeline progression.
- Use a single worker against a local/shared filesystem until those durability
  features are implemented and tested.
- Do not expose the unauthenticated demo web application publicly.


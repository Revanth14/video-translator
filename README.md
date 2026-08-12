# Video Translator

Customer-facing MVP for uploading a source video, translating/dubbing it, and
returning video, subtitle, and audio outputs.

## Local Web App

```bash
uv run dub-mvp web --no-open --port 8787
```

Open:

```text
http://127.0.0.1:8787
```

## Queued Web App

Use this when a GPU worker will process stages:

```bash
uv run dub-mvp web --runner queued --runs runs --no-open --port 8787
```

## GPU Worker

Process one queued stage:

```bash
uv run dub-mvp worker --runs runs --once
```

Run continuously:

```bash
uv run dub-mvp worker --runs runs --poll-seconds 5
```

## Deployment

- Copy `.env.example` to `.env` and fill provider credentials.
- Run `scripts/bootstrap-gpu.sh` on a fresh Ubuntu GPU droplet.
- Follow `docs/gpu-droplet-runbook.md`.
- See `docs/gpu-worker-contract.md` for the manifest and queue contract.

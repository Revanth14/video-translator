# GPU Droplet Runbook

This is the operator procedure for a single-host queued worker. Durable
architecture and known gaps live only in `docs/current-architecture.md`.

## Prerequisites

- An Ubuntu GPU host with a compatible NVIDIA driver
- Python 3.10
- The repository copied or cloned to `/srv/video-translator`
- A shared `runs/` directory visible to both web and worker processes
- Provider credentials configured outside version control
- Hugging Face read access to `ai4bharat/IndicF5`; `HF_TOKEN` is needed only
  when the checksum-verified model cache is not already present

## Bootstrap

From the repository root:

```bash
sudo PROJECT_DIR=/srv/video-translator scripts/bootstrap-gpu.sh
```

The bootstrap installs the lightweight application environment and a separate
GPU-qualified IndicF5 environment under `/opt/video-translator`. It syncs the
exact dependency lock, including TorchCodec, downloads pinned model/vocoder
revisions when missing, verifies their SHA-256 checksums, and prints the four
`VIDEO_TRANSLATOR_INDICF5_*` values the worker needs. Set
`INSTALL_INDICF5_RUNTIME=0` only for a core-only host that will never synthesize.

Copy `.env.example` to a protected deployment environment, fill required
credentials, and inject those values through the deployment's secret manager.
Do not bake `.env`, `HF_TOKEN`, source media, or voice references into an AMI.

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

1. Run `uv run dub-mvp preflight` with the printed IndicF5 variables and confirm
   FFmpeg, FFprobe, and the isolated IndicF5 runtime pass. The IndicF5 check
   imports TorchAudio and TorchCodec so a decoder mismatch fails before work.
2. Start the web application in queued mode.
3. Upload a short authorized video.
4. Run the worker once and verify the queued ingest stage is processed.
5. Inspect the run's `manifest.json` and confirm ingest completed.
6. Start the continuous worker only after the one-shot check succeeds.

## Deployment boundary

- This procedure configures one host and its local/shared filesystem; it does
  not create AWS Batch, S3 artifact storage, or remote conditional state.
- A sanitized AMI must contain model/runtime assets only. Delete and verify all
  authorized source, reference, and generated media before imaging the host.
- Do not expose the unauthenticated demo web application publicly.

# Dub MVP

An English-to-Hindi video dubbing MVP built as a resumable, segment-oriented
pipeline.

The current vertical slice provides:

- Validated run configuration.
- Atomic JSON run manifests.
- FFprobe media inspection.
- FFmpeg source-range extraction.
- Lossless 16 kHz mono working audio.
- Clear stage failures and resumable run state.

The product and sprint decisions live in
[`docs/dub-sprint-plan.md`](docs/dub-sprint-plan.md).

## Requirements

- Python 3.10
- FFmpeg and FFprobe
- `uv`

## Setup

```bash
uv sync
```

## Ingest a source segment

```bash
uv run dub-mvp ingest \
  --input /path/to/source.mp4 \
  --start 00:10:00 \
  --end 00:11:30 \
  --output runs
```

The command creates a timestamped run directory containing:

```text
runs/<run-id>/
  manifest.json
  metadata/
    ffprobe.json
  working/
    source_segment.mp4
    source_audio.wav
```

Inspect a run:

```bash
uv run dub-mvp show runs/<run-id>
```

## Tests

```bash
uv run pytest
```

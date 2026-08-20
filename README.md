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

## Run Status

Inspect the same durable status document used by the web app:

```bash
uv run dub-mvp status runs/<run-id>
```

It includes stage and work-item attempts, utterance progress, timings,
resources, reported cost, structured errors, and recent events.

## Duration Correction

Synthesis now measures every generated WAV and applies a bounded,
least-damaging duration policy before render. Raw TTS remains immutable and
reusable; correction attempts and verified outputs live under
`speech/duration/`. Unresolved timing violations are visible in status and are
rejected before render instead of receiving an unrecorded tempo adjustment.

Compact semantic rewriting is an injectable capability, not a silently added
provider. Benchmark and human review remain required before enabling one in a
production configuration.

## Rendering

Rendering creates a full-duration 48 kHz stereo audio bed, normalizes
loudness, limits peaks, preserves source-timeline silence, copies the video
stream, and validates the final MP4 with FFprobe plus a full FFmpeg decode.
Clean speech replacement is the default. Original-track ducking is explicit:

```bash
uv run dub-mvp render runs/<run-id> --composition duck_original
```

Every plan, subtitle, command history, WAV, MP4, and render report is
revisioned and checksum verified. Interrupted rendering reuses verified
intermediates.

## Voice References

IndicF5 is duration conditioned. Left to itself it predicts generated length
from the ratio of reference to target text measured in UTF-8 bytes. That
estimate breaks across scripts — Devanagari costs about 2.6 bytes per character
against Latin's one — so an English reference prompting Hindi over-predicted
duration by roughly that factor and the model padded the surplus with filler.

Synthesis therefore **pins generation to each utterance's timeline budget**
(`fix_duration`) instead of using that estimate. The utterance lands on its slot
by construction and the byte heuristic never runs.

Two constraints are enforced before GPU time is spent, because both are real
model limits:

- reference audio between 3 and 12 seconds, since F5-TTS clips anything longer
  during preprocessing and the model would not hear the clip that was measured
- reference plus target within the 25 s conditioning window

Everything else is **recorded, not enforced**. A reference transcript in a
different script than the target is expected in source-clone dubbing, where the
speaker's own English audio prompts Hindi output, so it is reported as a note
and a preflight warning. Speaking rates for both texts are recorded too, but no
threshold is applied to their ratio: a Latin letter and a Devanagari akshara are
not the same unit of speech, so comparing them would be an unmeasured heuristic.
Measured output duration and the duration-correction stage decide acceptance,
and cross-lingual voice quality is decided by listening.

## Benchmark

Before spending provider or GPU capacity on the qualifying long-form run, use
the strict benchmark readiness profile:

```bash
uv run dub-mvp preflight \
  --profile benchmark \
  --input-video evaluation/authorized-35-minute-source.mp4 \
  --voice-reference evaluation/voice-catalog.json \
  --target-language hi
```

The `voice_reference:prompt` check flags unusable reference durations, blocking
the benchmark profile while remaining a warning during local development. A
script mismatch is always advisory because English-reference → Hindi prompting
is the source-clone product and its quality must be verified by listening.

Unlike the default local preflight, this exits nonzero unless the 30–45 minute
input contract, FFmpeg, NVIDIA/CUDA runtime, WhisperX/OpenAI/IndicF5/Torch
modules, API credential, translation pricing, and consented voice catalog are
all present. It validates readiness; it does not call providers or spend GPU
capacity.

Aggregate the run's durable evidence without rerunning providers:

```bash
uv run dub-mvp benchmark runs/<run-id>
```

This writes `benchmark/*.json`, `benchmark/*.md`, and a human-review template.
After filling that template, include it with:

```bash
uv run dub-mvp benchmark runs/<run-id> --human-review review.json
```

Missing GPU, pricing, long-form, or human evidence is reported as
`not_measured`; it is never converted to zero or a release pass.

## Operator Recovery

Retry only selected stable utterance IDs and the artifacts downstream of them:

```bash
uv run dub-mvp retry \
  --run runs/<run-id> \
  --utterances 18,19,20 \
  --from synthesize
```

The command invalidates proof sidecars rather than deleting prior outputs,
preserves attempt history, records a verified retry report, and queues the
earliest affected stage. It refuses to invalidate active leased work.

## Release Readiness

Require a verified passing benchmark before treating a local run as releasable:

```bash
uv run dub-mvp release-check --run runs/<run-id> --target local
```

`--target aws` currently exits nonzero. The same worker executor has a core
container definition, but GPU/provider dependencies, remote conditional state,
S3 artifact transfer, interruption proof, and measured cloud cost are
deliberately blocked until a real long-form benchmark passes.

Language and training expansion are also evidence-gated:

```bash
uv run dub-mvp language-check \
  --run runs/<passing-hindi-run> \
  --candidate ta \
  --evaluation-set evaluation/ta.json

uv run dub-mvp research-check --decision research/decision.json
```

The admitted product language pair remains English → Hindi. A readiness pass
is evidence to make a separate reviewed registry change; it does not silently
enable a new language or train a model.

## Worker Container

The current container packages the core worker loop and FFmpeg runtime:

```bash
docker build -t dub-mvp-worker .
docker run --rm -v "$PWD/runs:/runs" dub-mvp-worker
```

It is not yet a benchmark-qualified GPU image and should not be represented as
an AWS deployment.

## Deployment

- Copy `.env.example` to `.env` and fill provider credentials.
- Run `scripts/bootstrap-gpu.sh` on a fresh Ubuntu Python 3.10 GPU host. It
  installs the exact isolated IndicF5 dependency lock, including TorchCodec,
  and checksum-verifies pinned model/vocoder artifacts before preflight.
- Provide `HF_TOKEN` during the first bootstrap when the qualified model cache
  is absent; never persist that token or authorized voice media in an image.
- Follow `docs/gpu-droplet-runbook.md`.
- See `docs/gpu-worker-contract.md` for the manifest and queue contract.

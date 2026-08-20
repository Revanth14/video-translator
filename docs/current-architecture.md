# Current Architecture

Describes the system as it stands. **Update this file in the same change that
alters the architecture** — a stale architecture doc is worse than none,
because agents and new contributors trust it.

Last verified: 2026-08-20 (296 tests passing, including process interruption,
real FFmpeg duration correction, full render/decode, killed-mux recovery,
selective retry, low-disk rejection, and release- and benchmark-readiness
gates).

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
`synthesize`, `render`, `benchmark`, `retry`, `release-check`,
`language-check`, `research-check`, `preflight`, `ui`, `web`, `worker`,
`status`, `show`.

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
Translation provider/network errors and malformed model responses are
retryable, bounded by `max_attempts`: model output is stochastic, so one bad
response must not kill a long run, while a systematically bad prompt still
terminates. Invalid translation *inputs* (missing segments file, corrupt
glossary, oversized utterance) raise `LocalizationError` and stay terminal.

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
Before expensive execution it measures free storage against a conservative
stage/source-size requirement. Insufficient disk raises a visible
`StorageCapacityError`; a leased worker treats it as transient and applies the
same bounded retry/backoff policy, while a direct operator invocation fails
terminally instead of starting work that cannot publish safely. Preflight
projects the same check for queued/running stages.

### Operator retry

`dub-mvp retry --run <run> --utterances <ids> --from <stage>` accepts stable
utterance IDs or their unique numeric suffixes. Supported retry boundaries are
`localize`, `synthesize`, and `render`. Translation retry expands one selected
utterance to its existing deterministic batch, because that batch is the
smallest independently verifiable localization unit. Synthesis retry
invalidates only selected raw speech/duration sidecars; unselected utterance
proof remains completed and reusable. Aggregate synthesis, render, benchmark,
and other genuinely downstream proof is invalidated as required.

Retry is a two-phase durable transition. A locked mutation first marks the
affected stages `INVALIDATED`, fencing the worker. Sidecars are changed from
`completed` to `invalid` without deleting output bytes. A verified
`metadata/retries/retry-<id>.json` audit artifact then records the decision,
and a second locked mutation queues the earliest stage while returning later
stages to `PENDING`. Attempt history is preserved and the explicit operator
action grants three additional bounded attempts. Active `RUNNING` work is
rejected without changing the manifest.

### Status and observability

`observability.build_run_status()` is the shared status contract. Both
`dub-mvp status <run>` (with `show` retained as an alias) and the customer job
API serialize that model, so stage state, attempts, progress, configuration,
timings, resources, cost, structured errors, and recent events cannot drift
between operator and creator views.

Manifest schema version 2 adds per-stage resource usage and structured
`RunError` records. Phase 8 manifests migrate on load: legacy error strings are
redacted and represented as `legacy_error`; absent resource measurements stay
unknown rather than being fabricated. Direct CLI stage execution and leased
worker execution both record attempt/event history, wall time, process CPU
deltas, and peak process RSS. Resource figures are process-level measurements,
not GPU telemetry or per-child accounting; that deeper benchmark data belongs
to Phase 12.

Translation-batch, speech-utterance, and duration-fit attempt files are
projected into the same status document, including failures, provider/model,
latency, and reported cost. Unreadable histories surface explicitly instead of
disappearing. Duration metrics add the primary/hard tolerance rates, unresolved
count, rewrite/review count, measured next-cue overruns, and the automated
timing-gate result. Render validation derives missing/duplicate utterance IDs
and the overlap count from the plan actually rendered instead of asserting
them, so the benchmark's drift gate tests measurements rather than constants.
`redact_sensitive_text` runs before any durable structured error or event
detail is written. It covers self-identifying key shapes (`sk-`, `hf_`, `ghp_`,
`AKIA`/`ASIA`), URL basic-auth credentials, and named secrets in prose, JSON,
YAML, env assignments, and query strings — including the quoted JSON form
(`{"api_key": "..."}`) that provider SDKs echo back inside exception text,
which is the most likely way a key would reach `manifest.json`. It is
idempotent, because errors are redacted on write and again when an older
manifest is migrated on read. Over-redaction is the intended failure mode.

Every successful manifest commit atomically regenerates
`events/run-events.jsonl`, an ordered structured projection of the authoritative
stage events. If that projection is missing, corrupt, or stale, a status read
rebuilds it from `manifest.json`. A real multi-process test verifies concurrent
manifest writers leave one valid complete projection.

Phase 12 benchmark paths are published into the manifest after the report and
sidecars are durable. Publishing identical paths aborts the mutation, so
re-reading a benchmark does not churn the manifest revision. Status exposes
the latest render-validation and benchmark release-gate result.

`release-check` independently verifies the benchmark JSON's completed status,
size, and checksum before trusting its embedded release result. Local release
passes only with `release_gate_status=passed`. AWS readiness additionally
requires a verified configuration snapshot and reports each absent remote
capability as `blocked`; a local benchmark cannot be mistaken for S3,
conditional remote state, incremental upload, a qualified GPU runtime, or
measured cloud cost.

## Artifacts

Run layout:

```text
runs/<run_id>/
├── manifest.json
├── events/         run-events.jsonl (recoverable manifest projection)
├── input/          source upload, glossary.json, translation-context.json,
│                   voice-reference.json, pipeline-config.json + sidecar
├── metadata/       ffprobe, whisperx_raw, transcript, segments,
│   └── retries/    verified operator-retry decisions
├── utterances/     dubbing_utterances.json, translation_segments.json,
│                   dubbing_utterances.meta.json
├── translation/    context snapshot, revisioned aggregate + sidecar,
│   └── batches/    request, attempts, revisioned result + sidecar per batch
├── speech/         revisioned aggregate, metrics, run record + sidecars,
│   ├── voice-maps/ persisted speaker map + sidecar
│   ├── utterances/ raw TTS attempts, revisioned WAV/result + sidecars
│   └── duration/   bounded fit attempts, candidates, results + sidecars
├── render/         command attempts, validated report + sidecars
├── benchmark/      JSON/Markdown report, review template + sidecars
├── working/        source_segment.mp4, source_audio.wav, revisioned dub WAV
├── subtitles/      revisioned Hindi SRT
└── outputs/        revisioned dubbed MP4
```

`artifacts.py` defines the reuse contract. An artifact is reusable only when
its sidecar exists, status is completed, the file exists, and size, SHA-256,
and input fingerprint all match. `path` is stored **relative to the run
directory** so a run survives being moved, copied, or uploaded to S3.

`fingerprint_inputs` canonicalizes JSON (sorted keys) and rejects bare
datetimes — a timestamp in a fingerprint means nothing is ever reusable again.

Changing a sidecar to `invalid` is the explicit operator-retry mechanism. It
is not file deletion: prior payloads remain inspectable, while every reuse
path rejects the invalid proof and writes a new revision.

### Configuration and expansion gates

Every admitted web job and CLI ingest snapshots the configured language pair,
translation/TTS provider and model identities, voice/glossary/context paths and
checksums, and duration/render policy fingerprints into verified
`input/pipeline-config.json`. The release registry currently admits only
`en→hi`; the HTTP API and ingest CLI reject other pairs before durable work is
queued.

`preflight` has two deliberately different profiles. The default `local`
profile requires FFmpeg/FFprobe and reports absent provider modules or
credentials as warnings, which keeps fixture and CPU-only development usable.
`preflight --profile benchmark` is the spending gate for the qualifying
long-form run: it exits nonzero unless an inspectable 30–45 minute input,
consented voice catalog, NVIDIA tooling, Torch-visible CUDA device,
WhisperX/OpenAI/IndicF5/Torch modules, OpenAI credential, and explicit current
translation token prices are all present. It performs no provider calls.

`language-check` does not enable a language. It verifies that the Hindi
baseline has a checksum-valid passing benchmark and that the candidate has a
source/target-aligned evaluation set with unique stable items. A separate
reviewed registry change is still required. `research-check` similarly parses
the explicit baseline X, measured bottleneck Y, objective Z, expected
improvement W, evaluation method E contract and requires its evaluation-set
file to exist; it does not start training.

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

### Translation contract

`LocalizationPipeline` partitions ordered dubbing utterances into deterministic
batches bounded by utterance count and source-character count. Each request
owns only its batch IDs; it receives the immediately preceding and following
source text as read-only context. Source/target language, tone, glossary,
named entities, terminology, prompt version, provider, model, and limits all
participate in the batch input fingerprint.

Each completed batch is written and fsynced immediately, then receives a
relative-path checksum/fingerprint sidecar. Restarting the `localize` stage
verifies and reuses valid batches and calls the provider only for missing,
failed, or corrupt work. Batch attempts are persisted independently; a real
process-death test verifies resume after the first batch without repeating it.
Forced or regenerated outputs use new revisions rather than overwriting a
verified artifact.

Provider output must return the expected batch ID, source language, target
language, and every owned utterance exactly once in the original order. Empty,
missing, duplicate, unknown, reordered, or wrong-language output is rejected.
Semantic translation is deliberately separate from later duration rewriting.

Translation metrics record batch reuse/regeneration, prompt version, tokens,
provider latency, attempts, and cost. Cost is recorded only when the provider
reports it or current per-million-token rates are explicitly configured;
otherwise `cost_status` is `pricing_unavailable`. The aggregate fingerprint,
provider, model, and cost are also committed to the localize `StageRecord`.

### Speech synthesis contract

`voice-reference.json` accepts the current `VoiceCatalog` schema (an ordered
`voices` list) and migrates the legacy single-`VoiceReference` shape at read
time. Each reference requires explicit consent metadata. Relative reference
audio paths resolve from the catalog directory; missing or empty reference
audio fails before synthesis.

#### IndicF5 duration conditioning

IndicF5 is duration conditioned. Without `fix_duration` it predicts generated
length from the reference-to-target text ratio **in UTF-8 bytes**, which is only
meaningful within one script: Devanagari costs about 2.6 bytes per character
against Latin's one, so an English reference prompting Hindi over-predicted
duration by roughly that factor and the model filled the surplus with filler.
Two evaluation samples measured 11.44 s for a ~4.5 s line, identical to the
millisecond across different chunkings — the length was a function of byte
counts, not speech.

`IndicF5Provider` therefore pins generation to each utterance's
`duration_budget_ms` (`indicf5_duration_plan`, policy
`fixed_timeline_budget_v1`). `fix_duration` is the whole conditioning window, so
the runtime restates it against the reference actually loaded after
`preprocess_ref_audio_text` and the generated portion equals the timeline slot.
The byte heuristic never executes.

Only two conditions are hard failures, because only two are model limits: a
reference outside 3–12 s (F5-TTS clips longer clips during preprocessing, so the
model would not hear the clip that was measured) and a reference-plus-target
window beyond 25 s.

Script pairing and speaking rate are **recorded, not enforced**. Source-clone
dubbing prompts Hindi output with the speaker's own English audio, so a
cross-script reference is the product rather than a defect; it is reported in
the synthesis notes and as a preflight `warn`. Unit rates for both texts are
recorded, but no threshold is applied to their ratio, because a Latin letter and
a Devanagari akshara are not the same unit of speech and a threshold on them
would be an unmeasured heuristic. Measured output duration and the
duration-correction stage decide acceptance; cross-lingual voice quality is
decided by listening (`scripts/evaluate-indicf5-crosslingual.py`).

#### IndicF5 provider text normalization

The Phase 1 listening gate passed native-Devanagari Hindi but failed the
technical case containing Latin `API key`, `environment variable`, and
`deployment script`. Translation remains the semantic and display artifact;
`IndicF5Provider` derives a separate pronunciation-only `tts_text` immediately
before inference. Policy `hindi_codeswitch_v1` converts only those evaluated
technical terms to Devanagari. It does not guess at unknown Latin names,
brands, URLs, or identifiers: their count is recorded in synthesis notes for
review.

The isolated request schema is version 5 and carries both the unchanged target
text and the provider `tts_text`, plus an explicit request ID and model
revision. The exact `tts_text` and normalization-policy version are part of
every IndicF5 utterance fingerprint. Evaluation can rerun one case with
`scripts/evaluate-indicf5-crosslingual.py --case technical` without replacing
the complete scoring sheet.

The Phase 1 source-clone gate passed on 2026-08-20. Four native-Devanagari
samples passed the fixed rubric on revision `d1e36b7`; the technical sample was
then regenerated with `hindi_codeswitch_v1` on revision `f121859` and passed the
same human thresholds. All five met the hard duration gate, and the reviewer
confirmed intelligibility of at least 4/5, acceptable voice similarity, no
severe filler or hallucination, and no clipped words. The consented audio and
checksum-verified review record remain under ignored `evaluation/`; they are
not source-controlled artifacts.

#### IndicF5 stage runtime

One `IndicF5Provider` child process belongs to one synthesis-stage execution.
The provider starts `indicf5_runtime.py --serve`, and the child loads and
compiles the model once before emitting its readiness record. Requests and
correlated responses are sequential NDJSON on stdin/stdout; every utterance
has a unique request ID. Provider/library output is redirected to stderr, which
the parent drains continuously on a dedicated thread so a full diagnostics
pipe cannot deadlock generation.

Stage cleanup runs in `finally`: close stdin, wait briefly, terminate, then kill
after another bounded wait. Cleanup errors are logged and cannot replace the
stage error being recorded. Unexpected child exit, malformed protocol output,
and timeout are retryable provider failures; missing models, incompatible
protocol/schema revisions, and invalid requests are permanent. A real child-
process test kills the runtime after the first completed utterance and proves
that the next stage attempt starts a new runtime, reuses the verified first WAV,
and requests only the unfinished utterance.

The real Phase 2 runtime gate passed on 2026-08-20 from revision `bd6e88a` on
one `g4dn.xlarge` (Tesla T4). Five sequential utterances used one child process:
14.21 s was spent in the one-time load/shutdown path and the complete run took
81.89 s. Per-utterance generation took 10.81, 11.15, 19.44, 11.46, and 14.40 s
for target windows of 1.6, 4.2, 11, 5, and 6.5 s. Peak measured GPU memory was
1,643 MiB, peak utilization reached 100%, and process max RSS was 3,424,468 KB.
Every output was within 0.7% of its target duration. Closing the stage left no
IndicF5 child or NVIDIA compute process. Human review passed all five for at
least 4/5 intelligibility, acceptable voice similarity, no severe filler or
hallucination, and no clipped words.

IndicF5 raw-speech and synthesis-configuration fingerprints include the model
revision, runtime protocol and implementation revision,
`fixed_timeline_budget_v1`, `single_batch_v1`, the text-normalization policy,
`configured_voice_catalog_v1`, and `preexisting_reference_audio_v1`. Changing
any of those generation inputs invalidates old speech proof. Runtime-script
paths and timestamps are deliberately excluded: a deployment path change does
not change audio semantics, while the explicit implementation revision does.

Before the first provider call, `SynthesisPipeline` deterministically assigns
speakers to the ordered catalog voices and persists a verified
`SpeakerVoiceMap`. Repeated utterances from one speaker therefore always use
the same voice. A catalog with one voice deliberately maps every speaker to
that voice; the system never invents an unconfigured provider voice.

When there are more speakers than voices, assignment wraps and two speakers
share a voice — audible to a listener but invisible to every automated gate.
`SpeakerVoiceMap.voice_collisions` derives the sharing from the assignments,
`SynthesisMetrics` reports `voice_count` and `voice_collision_count`, affected
utterances carry a note, and the run logs a warning. Sharing is allowed by
default because the web app supplies a single stock voice; pass
`--require-distinct-voices` (or `require_distinct_voices=True`) to fail before
the first provider call instead, which is what a benchmark run should do.

Each localized utterance owns an independent fingerprint, attempt history,
revisioned WAV, structured result, and checksum/fingerprint sidecars. The
fingerprint includes the utterance text/timing/revision, speaker, chosen voice
audio checksum, target language, provider, and model. IndicF5 fingerprints also
include its exact provider text and the complete generation-policy set above.
Restarting synthesis reuses each valid utterance independently and calls the
provider only for missing, failed, stale, or corrupt work. Forced and regenerated speech
always uses a new revision; a process-death test verifies that completed
utterances survive when the next provider call terminates the process.

Provider output is first written to a temporary WAV, decoded and fsynced, then
atomically promoted. Empty, missing, unreadable, invalid, checksum-mismatched,
and stale audio is never reusable. Duration, sample rate, channel count, and
sample width are measured from the promoted WAV; a provider-reported duration
is not authoritative. Synthesis metrics record calls, reuse/regeneration,
attempts, failures, latency, voice-map/configuration fingerprint, and measured
generated duration. The provider and fingerprint are committed to the
`synthesize` stage record.

### Duration-correction contract

Duration correction is part of `synthesize`, after immutable raw TTS and before
the synthesized aggregate is committed. It is deliberately not another stage:
the corrector needs the existing utterance, assigned voice, raw speech sidecar,
and provider capability, and verified raw speech remains reusable when a fit is
interrupted or its policy changes.

For every utterance the corrector measures
`generated_duration - duration_budget` and
`generated_duration / duration_budget`; provider-reported duration is never
trusted. It then tries, in order: accept, an optional conservative provider
rate/pause call, lossless leading/trailing artificial-pause trimming, bounded
pitch-preserving FFmpeg `atempo`, an optional compact semantic rewrite,
regeneration with the already assigned voice, a final bounded stretch, and an
explicit unresolved result. The default does **not** add a rewriting model:
`DurationRewriter` is injectable because project rules require benchmarking
before adding a provider/model. Required glossary terms and the rewriter's
meaning/term-preservation assertions are validated before regeneration, and
every rewrite still requires human review.

`DurationPolicy` fixes the primary tolerance at 10% or 250 ms, the hard limit
at 20%, mild tempo delta at 12%, provider-control/rewrite counts, and an overall
attempt cap. The complete policy, transformer, and optional rewriter identity
participate in the fit and synth-stage fingerprints. Every slow tactic is
preceded by a durable `running` attempt; a killed process leaves visible
interrupted work, raw TTS is reused, and the correction resumes. Failures are
redacted and recorded instead of silently skipped.

Each fit result embeds the selected verified audio sidecar and points to its
attempt history. Reuse verifies result and audio checksums/sizes, measured WAV
duration, input fingerprint, stable utterance timing/voice, and exact attempt
history. Corrupt correction artifacts are regenerated without repeating raw
TTS. Aggregate duration metrics report primary/hard pass percentages,
unresolved and human-review counts, attempts/provider calls, worst error, and
the automated provisional gate.

Source start timestamps never change, so start-time drift is zero **by
contract**. That is a structural property and is stated as one rather than
reported as a measured `0`, which a downstream gate would otherwise read as
evidence. The measurable neighbour risk is corrected audio running past the
next utterance's cue — an utterance can sit inside the hard tolerance and still
collide, because tolerance is measured against its own window, not the gap to
the next cue. `next_start_overrun_count` and `maximum_next_start_overrun_ms`
measure exactly that, and any overrun fails the automated timing gate, so the
problem surfaces before an expensive render refuses the plan.

The aggregate `SynthesizedSegment` schema is version 3 and carries raw/final
audio provenance, measured error/ratio, fit status/strategy, and review flag.
The loader still accepts unversioned Phase 0–6 aggregates as version 1 and
Phase 7–9 version-2 aggregates. Render retains its legacy tempo path for those
in-flight runs. For version 3 it never applies a second hidden correction, and
it rejects an unresolved or over-hard-limit utterance before FFmpeg composition.

### Composition and render contract

`RenderPolicy` makes composition explicit. `clean_replacement` is the default
first synchronization proof: it creates a full-timeline silent bed and places
only dubbed voices. `duck_original` keeps the original audio but reduces it to
the configured volume inside utterance windows; this can retain source
dialogue and must not be confused with dialogue/background separation. No
separation model or complex-overlap mechanism has been added without benchmark
evidence.

Every utterance is resampled to the configured 48 kHz stereo floating-point
layout, receives short edge fades, and is placed at its immutable source start.
Legacy synthesized artifacts receive their one bounded tempo correction here;
schema-v3 artifacts never receive a second hidden fit. Even a Phase 10
tolerance-accepted utterance is rejected if its real effective end would cross
the next utterance or final timeline. The full mix uses deterministic loudness
normalization and peak limiting, is trimmed to the requested run duration, and
is written as PCM WAV.

Muxing maps the original video and dubbed audio explicitly, copies the video
stream, encodes AAC audio, preserves source metadata, sets an explicit duration
and fast-start flag, and deliberately does not use `-shortest`. The old command
could truncate a video at the last utterance; the full-duration bed and explicit
duration close that defect.

Plans, subtitles, WAVs, MP4s, command histories, and the final render report are
revisioned artifacts with relative checksum/fingerprint sidecars. A command is
durably recorded as `running` before execution and closed afterward. After a
process death, the interrupted command is marked failed and valid intermediates
are found by their sidecars across revisions. Corrupt completed output is never
overwritten: a new revision is created while other verified intermediates are
reused.

Before completion the renderer probes source, WAV, and MP4; checks duration,
sample rate, channels, video codec/dimensions/frame rate; measures decoded peak;
and fully decodes the output video and audio with `-xerror`. A failed input or
validation contract is permanent; a tool execution failure remains retryable.
The render report is itself verified and is the authoritative reuse proof.

### Benchmark and human-evaluation contract

`dub-mvp benchmark <run>` reads the manifest and existing verified-stage
evidence; it never reruns a provider. Its revisioned JSON and Markdown reports
aggregate input media, ASR confidence and real-time factor, translation
tokens/retries/cost, synthesis runtime/reuse, median/p95 timing error, render
validation, per-stage attempts/resources/cost, storage, stable-ID integrity,
and speaker/voice consistency. A compact human-review template selects the
beginning/middle/end, every speaker, fast/slow speech, and available evidence
for names, terminology, rewrites, and low-confidence ASR. Reviewers explicitly
declare whether noise, music, or overlap is present and must cover those tags
when applicable.

Automated gates use `passed`, `failed`, `not_measured`, or `not_applicable`.
Unavailable GPU time/VRAM/cost and absent human scores remain `not_measured`,
never zero. A short fixture fails the 30–45 minute scope gate. Critical defects
remain separate from median scores, and critical mistranslation/name/omission
categories fail their own gate. Report reuse excludes benchmark output paths
from its fingerprint, so publishing a report does not invalidate itself;
changed stage evidence, artifacts, configuration, or human review does.

## Web behaviour

- The customer surface uses a responsive, content-first glass layout with
  translucent functional controls, high-contrast media surfaces, reduced-motion
  support, and a solid fallback when `backdrop-filter` is unavailable. Visual
  styling does not own progress or orchestration state. The visible language
  selector exposes only the release-enabled English→Hindi pair; unsupported
  candidates cannot be submitted from the product UI.
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
- Glossary, translation context, and voice reference are validated and
  persisted at job creation, before the pipeline reaches the stages that need
  them. Translation context has explicit tone, named-entity, and terminology
  fields.
- The browser **only submits and observes**. It never advances stages.
- Run identity persists in `localStorage` and the `?job=<run_id>` URL, so a
  reload or a later visit restores the run. A permanently missing job (400/404)
  releases the stored id; transient errors keep polling. An explicit URL id
  wins over stale local storage, and clearing the view preserves unrelated
  query parameters.
- Status polling uses one recursive timeout, scheduled only after the previous
  request completes. Rendered, failed, and cancelled runs do not schedule
  another request, including immediately after restoration. A response for a
  run cleared or replaced while its request was in flight is ignored.
- API responses are parsed through one guarded JSON boundary. A static server,
  proxy, or other endpoint returning HTML cannot leak a raw `Response.json()`
  exception into the interface; the selected file remains available and the
  page shows the correct `dub-mvp web` startup command.
- Creator retry is intentionally absent: a restored browser no longer owns the
  upload `File`, and retries must not become client-side stage orchestration.
  The stage endpoint remains an operator/debugging surface only; durable
  recovery is the server-side `dub-mvp retry` command.

## Known gaps

- IndicF5 Hindi code-switch normalization currently covers only the six Latin
  technical tokens proven by the Phase 1 evaluation. Unknown Latin tokens are
  measured and left unchanged; broader pronunciation handling needs its own
  evaluated term set before the lexicon expands.
- The real IndicF5 GPU gate covers one five-utterance stage, not the complete
  30–45 minute workload. Long-form thermal behavior, memory over many
  utterances, interruption/resume on Spot, and cost per source minute remain
  part of the full benchmark rather than being inferred from this short gate.
- `cli.py` still holds a manifest across pipeline execution (the pattern fixed
  in `runner.py`). Safe only because nothing else writes during a CLI run; it
  will conflict if an operator runs a stage while a worker holds a lease.
- Overlap detection is not implemented yet.
- The Phase 10 correction mechanism and provisional automated thresholds are
  implemented, but the 30–45 minute benchmark and human semantic review have
  not run. The optional compact-rewrite provider is intentionally unconfigured
  until that benchmark demonstrates it is needed and identifies a suitable
  model. Consequently, severe real-provider output can be surfaced as
  `unresolved` and will block render rather than being hidden.
- The Phase 12 aggregator is implemented, but this repository has not been
  given a real 30–45 minute provider run or completed human-review submission.
  GPU time, utilization/VRAM, and GPU/storage pricing are not instrumented, so
  the complete-reporting and release gates honestly remain `not_measured`.
- Phase 13 recovery mechanisms are implemented and covered for corruption,
  missing proof, lease fencing/reclamation, competing workers, bounded
  provider retries, supervisor faults, low disk, and real process death in
  localization, synthesis, duration fitting, and rendering. A real provider
  run is still required to exercise provider-specific timeout/rate-limit
  behavior and every major-stage kill scenario at long-form scale.
- Phase 14 is deliberately not complete. `Dockerfile` packages the core
  executor and FFmpeg but is not a qualified GPU container. The EC2 Phase 2
  image qualifies only the isolated IndicF5/Torch runtime; WhisperX/OpenAI,
  an S3 artifact backend, remote conditional state/fencing, incremental upload,
  AWS Batch resources, Spot-interruption proof, and measured end-to-end cloud
  cost are still absent. `release-check --target aws` makes those absences
  explicit blockers.
- Phase 15 is deliberately not complete. English→Hindi is the only admitted
  pair; no second language/provider/model or trained component has been added.
  Readiness contracts prevent expansion without a passing Hindi benchmark,
  candidate evaluation set, and explicit research evidence.
- Clean replacement and original-track ducking are implemented. Dialogue/
  background separation and complex overlap composition remain deliberately
  unimplemented until benchmark evidence justifies a model and its cost.
- Resource tracking currently covers process CPU, wall time, and peak RSS; it
  does not yet measure GPU utilization/VRAM or attribute child-process resource
  use precisely.
- Translation batches resume independently through verified artifacts, but
  they remain work items inside one leased `localize` stage; they are not
  independently scheduled across workers.
- The web app has no upload progress reporting; a large upload is silent until
  it completes.
- No authentication: a run id is not authorization. This is a controlled demo,
  not a public service.

## Environment

- `pyproject.toml` declares only the lightweight core (`pydantic` and `typer`).
  The isolated IndicF5 environment is separately pinned in
  `requirements-indicf5-gpu.txt`; it includes the GPU-qualified Torch,
  TorchAudio, TorchCodec, Transformers, and exact AI4Bharat Git revision.
  `scripts/bootstrap-gpu.sh` installs that environment and
  `scripts/cache-indicf5-models.py` downloads exact model/vocoder revisions and
  rejects checksum mismatches. WhisperX/OpenAI remain deployment-provided, and
  the current container intentionally captures the core executor and FFmpeg
  only.
- Python is pinned to `>=3.10,<3.11`.
- Translation cost estimation reads
  `VIDEO_TRANSLATOR_OPENAI_INPUT_USD_PER_MILLION` and
  `VIDEO_TRANSLATOR_OPENAI_OUTPUT_USD_PER_MILLION`. Rates are deliberately not
  hard-coded because provider pricing changes.
- Run the suite with `uv run python -m pytest` (plain `uv run pytest` may fail
  to spawn).
- Node.js is a **test-only** prerequisite: it runs `site/app.js` in the browser
  behaviour harness, the only executable coverage of the frontend. Without it
  those scenarios skip and the suite still reports success, so CI should set
  `VIDEO_TRANSLATOR_REQUIRE_NODE=1` to turn the skip into a failure.

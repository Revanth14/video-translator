# Agent Working Rules

Rules for any agent writing code in this repository.

**Every rule below exists because it was violated in this repo and the test
suite still passed.** They are not style preferences. Each one names the defect
that produced it, because a rule without its cause gets misapplied.

This file is about *how* to build, not what to build. The phase plan lives in
`docs/video-translator-final-build-plan.md`, which is intentionally untracked —
read it when present, and ask rather than guess when it is not.

---

## 1. The architecture you must not break

Three invariants outrank any convenience:

1. **The manifest is the authority.** Workers, threads, browser tabs, and web
   processes are disposable. State lives in `manifest.json`, never in a
   variable, a thread, or a client.
2. **One executor, many environments.** `LocalJobRunner` in-process and the
   external GPU worker run the *same* loop. Never fork a second orchestration
   path; if local and remote behave differently, that is a bug.
3. **Completed work is never redone.** Reuse requires proof (verified sidecar),
   not a file existing on disk.

---

## 2. Never hold durable state across slow work

**Defect:** `_run_stage` loaded the manifest, ran a multi-minute pipeline, then
saved. Any concurrent write (a heartbeat, a cancellation) made the final save
throw. Worse, the failure handler *also* saved, so it threw inside `except` and
destroyed the original error.

**Rules:**
- Every read-modify-write on the manifest goes through `mutate_manifest()`,
  which loads and writes inside one lock. Keep the callback short and pure.
- Never pass a `RunManifest` into long-running work and write it afterwards.
  Read the values you need, drop the object, re-open a mutation to record the
  result.
- A failure path must never be able to fail for the same reason as the
  operation it is reporting.

## 3. Data models are not behaviour

**Defect:** `worker_id`, `lease_generation`, `heartbeat_at`, and
`lease_expires_at` were all written to the manifest, and nothing renewed,
reclaimed, or checked them. The lease looked implemented and enforced nothing.

**Rules:**
- Adding a field is not a feature. Ship the code that *reads* it in the same
  change, plus a test where the field's value changes an outcome.
- If an enum has members nothing can produce (`OverlapStatus.CONFIRMED` today),
  say so in the docstring or don't add them yet.

## 4. Terminal states must actually terminate

**Defect:** a stage that exhausted `max_attempts` was left `QUEUED`.
`claim_job` declined it, `find_next_queued_job` returned it again, and the
worker spun on it forever — starving every other run. The test asserted the
stage stayed `QUEUED`, so it passed while the system was livelocked.

**Rules:**
- Refusing to process work is not the same as recording that it is finished.
  Move it to a terminal state.
- After changing scheduling, assert **liveness**, not just the return value:
  "the next scan finds other work" / "the queue drains".
- One bad run must never block the runs behind it. Contain faults per unit of
  work (`unavailable` set in `run_worker_once`), not per pass.

## 5. Supervisors never exit

**Defect:** `run_worker_loop` had no exception guard. A `PermissionError` on
one run directory killed the loop. In the web app that is a daemon thread, so
the server kept answering requests while every job silently stopped forever.

**Rules:**
- Any loop that supervises other work catches broadly, logs, backs off, and
  continues.
- Any stage executor catches broadly and *records the failure* — never leave a
  stage in `RUNNING` with no explanation.
- Distinguish **permanent** from **transient** failure everywhere, backend and
  frontend. Transient retries with backoff; permanent goes terminal and visible.
  (The UI polled a deleted job once a second forever because it treated a 400
  the same as a network blip.)

## 6. Artifacts must prove themselves

**Rules:**
- `path` in any sidecar is **relative to the run directory**. Absolute paths
  break the moment a run is moved, copied, or uploaded to S3 — which Phase 14
  requires. Use `relative_artifact_path()`.
- Reuse requires all of: sidecar exists, status completed, file exists, size
  matches, checksum matches, input fingerprint matches. `path.exists()` is
  never sufficient.
- **Input fingerprints contain only things that should force regeneration.**
  Never a timestamp — it differs every call and no artifact is ever reusable
  again. `fingerprint_inputs` rejects bare datetimes; it cannot catch one
  nested inside a model, so check yourself.
- Write order: write payload → fsync → validate → atomic `os.replace` → mark
  complete. A crash must leave work unverifiable, never falsely complete.

## 7. Changing a limit means changing the mechanism behind it

**Defect:** Phase 4 removed the 90-second cap so full videos could be
processed. The upload path still reads the entire body into memory and copies
it twice (measured: **4× amplification**, 100 MB upload → ~400 MB RAM). The
policy changed; the mechanism that made the policy safe did not. The stated
exit gate was unreachable, and tests missed it because the fake ingestor
receives `b"video-bytes"`.

**Rules:**
- When removing a limit, ask what the limit was protecting, and verify that
  thing still holds at the new scale.
- Verify an exit gate with input of realistic *size*, not just realistic shape.
- If a fake hides the exact constraint you changed, the fake is not evidence.

## 8. Never silently drop or diverge

**Rules:**
- Do not duplicate logic that must stay identical across modules. Word
  flattening exists in both `transcribe.py` and `utterances.py`; if they ever
  drift, `source_word_indexes` points at the wrong words and nothing fails
  loudly. Import the one implementation.
- Skipping malformed input silently (`if index in words_by_index`) is data
  loss. Raise, or record the omission in the manifest.
- Preserve stable IDs (`segment_id`, `utterance_id`) across every stage. They
  are the only handle for selective retry.

## 9. Schema changes must not strand in-flight runs

Inserting `segment` into `STAGE_ORDER` would have made every downstream stage
permanently unclaimable for existing manifests, because `_dependencies_complete`
walks all prior stages. `ensure_pipeline_stages` back-fills it. **This is the
standard to meet, not an exception.**

**Rules:**
- Any change to `PIPELINE_STAGE_NAMES`, `StageRecord`, or an artifact schema
  ships with a migration for manifests written by the previous build.
- Test the migration against hand-written legacy JSON, covering both a run that
  stopped before the new stage and one that already ran past it.
- Bump `schema_version` when the shape changes.

## 10. Keep the architecture doc true

**Defect:** `docs/current-architecture.md` and the Phase 0 assumptions doc were
accurate when written, then the next phases fixed exactly what they described.
Within days they asserted the *inverse* of reality — "no lease, heartbeat, or
fencing token" while all three existed. A stale doc is worse than no doc,
because agents trust it.

**Rules:**
- `docs/current-architecture.md` is part of the architecture, not a report about
  it. A change to the stage graph, execution model, durability contract,
  artifact layout, or web behaviour is **not complete** until that file matches.
- Do not add "current state" prose anywhere else; there is one such file.
- Record real gaps in its *Known gaps* section rather than starting a new
  document. Snapshots rot; a single living document can be corrected.
- This is the same failure this codebase keeps hitting: a recorded state that
  drifts from the real state, which everything downstream then trusts.

## 11. Naming must match behaviour

`find_next_queued_job` mutates durable state (it queues the next ready stage).
A `find_*` that writes will mislead the next reader. Name functions for what
they do, or split the query from the command.

---

## 12. Verification: what "done" means

**Passing tests is not evidence that a durability feature works.** 85 tests
passed while the manifest design was fundamentally incompatible with
heartbeating, and every expensive dependency is mocked.

Before claiming a task is complete:

- **Exercise the real failure.** Kill the process, revoke permissions, corrupt
  the artifact, expire the lease. Do not assert a fault is handled — cause it.
- **Cross-process, not just in-process.** Unit tests all run in one process, so
  `flock` is never exercised. Concurrency claims need real processes. (The CAS
  was verified with 8 processes × 25 writes: 200/200 recorded, no lost updates.)
- **Measure resource claims.** Memory, time, and cost assertions need numbers
  from `tracemalloc`/timing, not reasoning.
- **Re-run the check after fixing.** The first worker-loop fix stopped the loop
  from dying but left the poisoned run starving every other run — only visible
  because the verification was run again. A fix is not done until the original
  reproduction passes.
- **Report honestly.** If a gate is not met, say so plainly and say why. Never
  describe intended behaviour as achieved behaviour.

## 13. Tests

- A test that pins current behaviour is not a test of correct behaviour. When
  fixing a bug, check whether an existing test asserts the bug — and change it
  deliberately, calling that out.
- Write the test so it fails for the right reason. Prefer `monkeypatch` over
  `chmod` for fault injection: running as root silently bypasses permission
  bits and the test passes for the wrong reason.
- Keep expensive providers injectable and mocked; keep real-provider tests
  separate.
- Cover: state transitions, claim/lease/fencing, retry and backoff bounds,
  fingerprint and checksum invalidation, resume-after-kill, and migrations.

## 14. Scope and reporting

- Fix what was asked. If you find an adjacent defect, fix it only if it is in
  code you are already changing — and say that you did.
- Flag what you deliberately left alone and why.
- Prefer extending the existing architecture over rewriting it. The resumable
  manifest, provider abstractions, and artifact contracts are load-bearing.
- Match the surrounding code: `from __future__ import annotations`, Pydantic
  models for contracts, atomic writes via temp + `os.replace`, keyword-only
  arguments for anything non-obvious, comments that explain *why*.
- Do not add providers, services, models, or AWS components without a measured
  problem. Benchmark first.

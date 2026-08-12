# GPU Worker Contract

This document records the current queued-worker boundary and the requirements
for its durable replacement.

## Current contract

- The web process creates a run and marks a requested stage `queued`.
- Queue events are appended to `metadata/job-queue.jsonl`.
- The worker scans `*/manifest.json` files for queued stages.
- A synchronous `LocalJobRunner` executes the selected stage.
- Stage outputs and final state are written back to the run manifest.

The web and worker must share the same run directory in the current deployment
scaffold.

## Current safety boundary

The current load/check/save claim is not a safe multi-worker compare-and-set
operation. Run only one worker against a shared filesystem.

## Target contract

The durable state backend will expose operations equivalent to:

```text
get_run
compare_and_claim
renew_lease
complete
fail
cancel
reclaim_stale
```

Every claim will include a worker identity, heartbeat, expiration time, and
monotonically increasing lease generation. Completion from an old generation
will be rejected so a stale worker cannot overwrite reclaimed work.

Artifacts will be reusable only when their sidecar validates status, path,
checksum, schema, and input fingerprint.


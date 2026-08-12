# GPU Worker Contract

This document records the current queued-worker boundary and the requirements
for its durable replacement.

## Current contract

- The web process persists all required inputs, creates a run, and queues ingest.
- Queue events are appended to `metadata/job-queue.jsonl`.
- The worker scans `*/manifest.json` files and advances the dependency chain:
  `ingest -> transcribe -> segment -> localize -> synthesize -> render`.
- A synchronous `LocalJobRunner` executes the selected stage.
- Stage outputs and final state are written back to the run manifest.

The web and worker must share the same run directory in the current deployment
scaffold.

## Current safety boundary

Manifest mutations use a per-run file lock and revision check. Claims include a
worker identity, expiring lease, heartbeat, and monotonically increasing fencing
generation. Expired work can be reclaimed, and stale workers cannot commit.

This safety applies to workers sharing one local POSIX filesystem. Network file
systems and distributed deployments still need a durable state backend with
real conditional writes.

## Distributed target contract

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

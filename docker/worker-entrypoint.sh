#!/bin/sh
set -eu

exec dub-mvp worker \
    --runs "${DUB_MVP_RUNS_DIRECTORY:-/runs}" \
    --poll-seconds "${DUB_MVP_POLL_SECONDS:-1}" \
    "$@"

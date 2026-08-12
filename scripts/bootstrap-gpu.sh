#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/srv/video-translator}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_DIR/runs}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:-}"

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
elif [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  echo "Run as root or install sudo before bootstrapping." >&2
  exit 1
fi

echo "Installing system dependencies..."
$SUDO apt-get update
$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  pkg-config \
  python3-dev

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project directory does not exist: $PROJECT_DIR" >&2
  echo "Copy or clone the repository there, then rerun this script." >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$RUNS_DIR"

echo "Syncing Python environment..."
if [ -n "$UV_SYNC_ARGS" ]; then
  uv sync $UV_SYNC_ARGS
else
  uv sync
fi

echo "Running preflight..."
uv run dub-mvp preflight || true

echo "Bootstrap complete."
echo "Worker command:"
echo "  uv run dub-mvp worker --runs $RUNS_DIR --poll-seconds 5"

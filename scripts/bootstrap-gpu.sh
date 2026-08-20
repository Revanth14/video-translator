#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/srv/video-translator}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_DIR/runs}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:-}"
INSTALL_INDICF5_RUNTIME="${INSTALL_INDICF5_RUNTIME:-1}"
INDICF5_ENV_DIR="${INDICF5_ENV_DIR:-/opt/video-translator/indicf5-venv}"
INDICF5_MODEL_CACHE="${INDICF5_MODEL_CACHE:-/opt/video-translator/model-cache}"
INDICF5_LOCK="${INDICF5_LOCK:-$PROJECT_DIR/requirements-indicf5-gpu.txt}"

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
  if [ "$(id -u)" -eq 0 ]; then
    UV_BIN_DIR="${UV_BIN_DIR:-/usr/local/bin}"
  else
    : "${HOME:?HOME must be set when bootstrap-gpu.sh is not run as root}"
    UV_BIN_DIR="${UV_BIN_DIR:-$HOME/.local/bin}"
  fi
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" sh
  export PATH="$UV_BIN_DIR:$PATH"
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

if [ "$INSTALL_INDICF5_RUNTIME" = "1" ]; then
  if [ ! -f "$INDICF5_LOCK" ]; then
    echo "IndicF5 lock file does not exist: $INDICF5_LOCK" >&2
    exit 1
  fi
  echo "Syncing the isolated, GPU-qualified IndicF5 environment..."
  if [ ! -x "$INDICF5_ENV_DIR/bin/python" ]; then
    uv venv --python 3.10 "$INDICF5_ENV_DIR"
  fi
  uv pip sync --python "$INDICF5_ENV_DIR/bin/python" "$INDICF5_LOCK"

  export VIDEO_TRANSLATOR_INDICF5_PYTHON="$INDICF5_ENV_DIR/bin/python"
  export VIDEO_TRANSLATOR_INDICF5_MODEL_ROOT="$INDICF5_MODEL_CACHE/indicf5-artifacts"
  export VIDEO_TRANSLATOR_INDICF5_VOCODER_ROOT="$INDICF5_MODEL_CACHE/vocos-mel-24khz"
  export VIDEO_TRANSLATOR_INDICF5_MODEL_REVISION="ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
  "$INDICF5_ENV_DIR/bin/python" "$PROJECT_DIR/scripts/cache-indicf5-models.py"
fi

echo "Running preflight..."
uv run dub-mvp preflight || true

echo "Bootstrap complete."
if [ "$INSTALL_INDICF5_RUNTIME" = "1" ]; then
  echo "Set these variables in the worker environment:"
  echo "  VIDEO_TRANSLATOR_INDICF5_PYTHON=$VIDEO_TRANSLATOR_INDICF5_PYTHON"
  echo "  VIDEO_TRANSLATOR_INDICF5_MODEL_ROOT=$VIDEO_TRANSLATOR_INDICF5_MODEL_ROOT"
  echo "  VIDEO_TRANSLATOR_INDICF5_VOCODER_ROOT=$VIDEO_TRANSLATOR_INDICF5_VOCODER_ROOT"
  echo "  VIDEO_TRANSLATOR_INDICF5_MODEL_REVISION=$VIDEO_TRANSLATOR_INDICF5_MODEL_REVISION"
fi
echo "Worker command:"
echo "  uv run dub-mvp worker --runs $RUNS_DIR --poll-seconds 5"

#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi bootstrap script.
# Run this from repository root after git clone.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
INSTALL_API_DEPS="${INSTALL_API_DEPS:-1}"

echo "[setup] repo_dir=$REPO_DIR"
echo "[setup] python_bin=$PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup][error] $PYTHON_BIN not found. Install Python 3 first."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/requirements.txt"

if [[ "$INSTALL_API_DEPS" == "1" && -f "$REPO_DIR/requirements-api.txt" ]]; then
  python -m pip install -r "$REPO_DIR/requirements-api.txt"
fi

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/reports" "$REPO_DIR/artifacts"

echo "[setup] done"
echo "[setup] next:"
echo "  source $VENV_DIR/bin/activate"
echo "  python scripts/pi_preflight.py --api-base-url http://<tailscale-host-or-ip>:8000"

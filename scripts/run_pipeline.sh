#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot pipeline runner for macOS / Linux.
#
#   1. creates a .venv (if missing)
#   2. installs requirements
#   3. runs the full training pipeline
#   4. runs the unit tests
#
# Usage:   bash scripts/run_pipeline.sh
# -----------------------------------------------------------------------------
set -e

# Resolve project root (directory above this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# 1. Virtual environment
if [ ! -d ".venv" ]; then
  echo "[run_pipeline] Creating virtual environment…"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Dependencies
echo "[run_pipeline] Installing dependencies…"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# 3. Training
echo "[run_pipeline] Training model…"
python -m src.training.train

# 4. Tests
echo "[run_pipeline] Running unit tests…"
pytest -q

echo "[run_pipeline] Done. Results are in ./results and ./models_saved"

#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Download the Kaggle dataset with the Kaggle CLI.
#
# Requirements:
#   * `pip install kaggle`  (already in requirements.txt)
#   * A Kaggle API token saved to ~/.kaggle/kaggle.json
#     See https://github.com/Kaggle/kaggle-api#api-credentials
#
# Usage:  bash scripts/download_data.sh
# ----------------------------------------------------------------------------
set -e

DATASET="shashwatwork/phishing-dataset-for-machine-learning"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
DEST="$PROJECT_ROOT/data/raw"

mkdir -p "$DEST"

echo "[download_data] Fetching $DATASET from Kaggle…"
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

echo "[download_data] Files now in $DEST :"
ls -la "$DEST"

"""
Command-line predictor.

Two modes:

1. ``--url "https://example.com"`` – extract lexical features from
   a single URL and print the prediction.

2. ``--csv path/to/features.csv`` – load a CSV that already contains
   the 48 Kaggle feature columns, run the model, and save predictions
   next to the input file.

Examples
--------
Single URL::

    python -m src.predict --url "http://paypal.com.login-secure-update.tk/"

Batch CSV::

    python -m src.predict --csv data/raw/Phishing_Legitimate_full.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import (
    extract_features_from_url,
    get_feature_column_names,
)
from src.utils import get_logger, load_config, project_root

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_trained_model():
    """Load the trained model + feature-column list from disk."""
    cfg = load_config()
    models_dir = project_root() / cfg["paths"]["models_dir"]
    model_path = models_dir / "phishing_rf_model.joblib"
    feats_path = models_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}.  "
            "Run `python -m src.training.train` first."
        )

    model = joblib.load(model_path)
    features = json.loads(feats_path.read_text())
    logger.info("Loaded model from %s", model_path)
    return model, features


# ----------------------------------------------------------------------
# Prediction modes
# ----------------------------------------------------------------------
def predict_single_url(url: str) -> dict:
    """Predict phishing probability for a single URL string."""
    model, feature_cols = _load_trained_model()
    feat_dict = extract_features_from_url(url)
    X = pd.DataFrame([feat_dict], columns=feature_cols)
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    return {
        "url":         url,
        "prediction":  "phishing" if label == 1 else "legitimate",
        "probability": proba,
    }


def predict_csv(csv_path: Path) -> Path:
    """
    Score a CSV that contains the pre-extracted feature columns.

    Parameters
    ----------
    csv_path : Path
        Path to a CSV with all 48 Kaggle feature columns.
    """
    model, feature_cols = _load_trained_model()
    df = pd.read_csv(csv_path)

    # Keep only the required feature columns (ignore id, label, etc.)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # For columns that require HTML scraping, fill with 0.0.
        logger.warning(
            "%d columns missing from input – filling with 0.0: %s",
            len(missing), missing[:5],
        )
        for c in missing:
            df[c] = 0.0

    X = df[feature_cols]
    df["phishing_probability"] = model.predict_proba(X)[:, 1]
    df["prediction"] = (df["phishing_probability"] >= 0.5).astype(int)
    df["prediction_label"] = df["prediction"].map({0: "legitimate", 1: "phishing"})

    out_path = csv_path.with_name(csv_path.stem + "_predictions.csv")
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d predictions to %s", len(df), out_path)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict phishing URLs using the trained model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url", type=str,
        help="A single URL to classify.",
    )
    group.add_argument(
        "--csv", type=Path,
        help="CSV file with the 48 Kaggle feature columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.url:
        result = predict_single_url(args.url)
        print(json.dumps(result, indent=2))
    else:
        predict_csv(args.csv)


if __name__ == "__main__":
    main()

"""
End-to-end training driver.

Usage
-----
From the project root::

    python -m src.training.train

The script:

1. loads the raw CSV,
2. cleans and splits the data,
3. engineers ratio features,
4. fits a ``RandomForestClassifier`` pipeline,
5. evaluates it on train, validation and test sets,
6. persists the trained model and evaluation artefacts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.evaluation.evaluator import evaluate_model, save_evaluation_artifacts
from src.feature_engineering import (
    engineer_features,
    get_feature_column_names,
)
from src.models import build_model
from src.preprocessing import clean_dataframe, load_raw_dataset, split_data
from src.utils import get_logger, load_config, project_root
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""
    cfg = load_config()
    lbl_col = cfg["columns"]["label_column"]
    feature_cols = get_feature_column_names()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")

    X = df[feature_cols].copy()
    y = df[lbl_col].astype(int).copy()
    return X, y


def run_training(seed: int | None = None) -> dict:
    """
    Execute the full training pipeline.

    Parameters
    ----------
    seed : int, optional
        Override the random seed defined in the config.

    Returns
    -------
    dict
        Summary metrics (useful for scripting).
    """
    t_start = time.time()
    cfg = load_config()
    if seed is not None:
        cfg["model"]["random_state"] = seed
        cfg["split"]["random_state"] = seed

    # ---------- 1.  Load & clean ------------------------------------
    df = load_raw_dataset()
    df = clean_dataframe(df)
    df = engineer_features(df)

    # ---------- 2.  Split -------------------------------------------
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = _prepare_xy(train_df)
    X_val,   y_val   = _prepare_xy(val_df)
    X_test,  y_test  = _prepare_xy(test_df)

    # ---------- 3.  Build & fit model -------------------------------
    model = build_model()
    logger.info("Fitting model on %d samples…", len(X_train))
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start
    logger.info("Training finished in %.2f seconds", fit_time)

    # ---------- 4.  Cross-validation on train set -------------------
    logger.info("Running %d-fold cross-validation…", cfg["runtime"]["cv_folds"])
    skf = StratifiedKFold(
        n_splits=cfg["runtime"]["cv_folds"],
        shuffle=True,
        random_state=cfg["split"]["random_state"],
    )
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=skf, scoring="f1", n_jobs=-1,
    )
    logger.info(
        "CV F1: mean=%.4f  std=%.4f", cv_scores.mean(), cv_scores.std()
    )

    # ---------- 5.  Evaluate on each split --------------------------
    results = {
        "train": evaluate_model(model, X_train, y_train, "train"),
        "val":   evaluate_model(model, X_val,   y_val,   "val"),
        "test":  evaluate_model(model, X_test,  y_test,  "test"),
        "cv_f1_mean":  float(cv_scores.mean()),
        "cv_f1_std":   float(cv_scores.std()),
        "training_time_seconds": round(fit_time, 3),
    }

    # ---------- 6.  Persist artefacts -------------------------------
    models_dir = ensure_dir(project_root() / cfg["paths"]["models_dir"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    model_path = models_dir / "phishing_rf_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    # Save the feature column list so the predictor knows the order.
    with (models_dir / "feature_columns.json").open("w") as fh:
        json.dump(get_feature_column_names(), fh, indent=2)

    # Summary metrics
    with (results_dir / "metrics_summary.json").open("w") as fh:
        json.dump(results, fh, indent=2, default=float)

    save_evaluation_artifacts(model, X_test, y_test, results_dir)

    total = time.time() - t_start
    logger.info("Pipeline completed in %.2f seconds", total)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phishing URL Detection model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_training(seed=args.seed)
    print(json.dumps(summary, indent=2, default=float))

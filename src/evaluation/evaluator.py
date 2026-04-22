"""
Evaluation utilities: metrics + figures.

All figures are saved as PNG files in ``results/`` so they can be
embedded directly into the technical report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # no GUI backend – safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------
def evaluate_model(model, X, y, split_name: str) -> Dict:
    """
    Compute key classification metrics for a single split.

    Parameters
    ----------
    model : sklearn estimator
    X : array-like features
    y : array-like labels
    split_name : str
        Used only for logging / book-keeping.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    t0 = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    inference_time = time.time() - t0

    metrics = {
        "split":        split_name,
        "n_samples":    int(len(y)),
        "accuracy":     float(accuracy_score(y, y_pred)),
        "precision":    float(precision_score(y, y_pred, zero_division=0)),
        "recall":       float(recall_score(y, y_pred, zero_division=0)),
        "f1":           float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc":      float(roc_auc_score(y, y_proba)),
        "inference_time_seconds": round(inference_time, 4),
    }
    logger.info(
        "%-5s | acc=%.4f  f1=%.4f  roc_auc=%.4f",
        split_name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
    )
    return metrics


# ----------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------
def _save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_roc_curve(y_true, y_proba, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pr_curve(y_true, y_proba, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve – Test Set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_feature_importance(model, feature_names, out_path: Path) -> None:
    """
    Feature-importance bar chart. Assumes a scikit-learn Pipeline
    whose last step has ``feature_importances_``.
    """
    # Walk to the last estimator in the pipeline.
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "feature_importances_"):
        logger.warning("Model has no feature_importances_; skipping plot.")
        return

    importances = estimator.feature_importances_
    order = np.argsort(importances)[::-1][:20]   # top 20
    top_features = np.array(feature_names)[order]
    top_values = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_values, y=top_features, orient="h", ax=ax)
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_evaluation_artifacts(model, X_test, y_test, out_dir: Path) -> None:
    """
    Save confusion matrix, ROC curve, PR curve, feature importance,
    and classification report to ``out_dir``.
    """
    out_dir = ensure_dir(out_dir)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    _save_confusion_matrix(y_test, y_pred,   out_dir / "confusion_matrix.png")
    _save_roc_curve(y_test,  y_proba,        out_dir / "roc_curve.png")
    _save_pr_curve(y_test,   y_proba,        out_dir / "pr_curve.png")
    _save_feature_importance(
        model, list(X_test.columns), out_dir / "feature_importance.png"
    )

    # Full classification report (human-readable)
    report = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"], digits=4,
    )
    with (out_dir / "classification_report.txt").open("w") as fh:
        fh.write(report)

    logger.info("Saved evaluation artefacts to %s", out_dir)

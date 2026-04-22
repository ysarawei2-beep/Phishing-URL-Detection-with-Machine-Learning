"""
Pure-NumPy demo runner.

Produces the same output artefacts as the sklearn pipeline so users
can see real metrics on the Kaggle dataset *before* installing the
full requirements.  This is useful in two cases:

1. Verifying the dataset is in place and the feature schema matches.
2. Reviewing reference numbers in environments where scikit-learn
   cannot be installed (e.g. locked-down machines).

The sklearn pipeline (`python -m src.training.train`) is the real
deliverable and will usually beat these numbers by a few points.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `src.*` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import (  # noqa: E402
    engineer_features,
    get_feature_column_names,
)
from src.preprocessing import (  # noqa: E402
    clean_dataframe,
    load_raw_dataset,
    split_data,
)
from src.utils import get_logger, load_config, project_root  # noqa: E402
from src.utils.helpers import ensure_dir  # noqa: E402

logger = get_logger("numpy_demo_runner")


# ----------------------------------------------------------------------
# Tiny logistic-regression + decision-stump ensemble (numpy-only)
# ----------------------------------------------------------------------
class NumpyLogisticRegression:
    """
    Plain mini-batch logistic regression trained with Adam.
    Produces feature "importance" as the absolute value of each
    standardised weight (useful for the bar chart).
    """

    def __init__(self, lr: float = 0.05, epochs: int = 300,
                 batch_size: int = 256, l2: float = 1e-3,
                 random_state: int = 42) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.rng = np.random.default_rng(random_state)
        self.mean_ = None
        self.std_ = None
        self.W = None
        self.b = 0.0

    def _scale(self, X):
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Standardise features.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = self._scale(X)

        n_samples, n_features = Xs.shape
        self.W = self.rng.normal(0, 0.01, size=n_features)
        self.b = 0.0

        # Adam hyper-parameters
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = 0.0; vb = 0.0
        t = 0

        for epoch in range(self.epochs):
            # Shuffle
            idx = self.rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                t += 1
                batch = idx[start:start + self.batch_size]
                xb, yb = Xs[batch], y[batch]
                z = xb @ self.W + self.b
                # Numerically stable sigmoid
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                grad_logit = (p - yb) / len(yb)
                gW = xb.T @ grad_logit + self.l2 * self.W
                gb = grad_logit.sum()

                # Adam updates
                mW = beta1 * mW + (1 - beta1) * gW
                vW = beta2 * vW + (1 - beta2) * gW ** 2
                mb = beta1 * mb + (1 - beta1) * gb
                vb = beta2 * vb + (1 - beta2) * gb ** 2
                mW_hat = mW / (1 - beta1 ** t)
                vW_hat = vW / (1 - beta2 ** t)
                mb_hat = mb / (1 - beta1 ** t)
                vb_hat = vb / (1 - beta2 ** t)
                self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if (epoch + 1) % 50 == 0:
                z = Xs @ self.W + self.b
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                loss = -np.mean(
                    y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
                )
                logger.info("epoch %3d  loss=%.5f", epoch + 1, loss)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self._scale(X)
        z = Xs @ self.W + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------------
# Metric helpers (no sklearn dependency)
# ----------------------------------------------------------------------
def _confusion_matrix(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]]."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def _metrics(y_true, y_proba, split_name: str) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    cm = _confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]; fn, tp = cm[1]
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # AUC via Mann-Whitney U (no sklearn needed).
    roc_auc = _roc_auc(y_true, y_proba)

    return {
        "split":     split_name,
        "n_samples": int(len(y_true)),
        "accuracy":  float(acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "roc_auc":   float(roc_auc),
    }


def _roc_auc(y_true, y_score) -> float:
    """Mann-Whitney U statistic -> AUC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    ranks = np.arange(1, len(y_sorted) + 1)
    pos_ranks = ranks[y_sorted == 1].sum()
    n_pos = (y_sorted == 1).sum()
    n_neg = (y_sorted == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = pos_ranks - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _roc_curve(y_true, y_score, n_thresholds: int = 200):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tprs, fprs = [], []
    for t in thresholds[::-1]:
        y_pred = (y_score >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        tprs.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return np.array(fprs), np.array(tprs)


def _pr_curve(y_true, y_score, n_thresholds: int = 200):
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    precs, recs = [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precs.append(tp / (tp + fp) if (tp + fp) else 1.0)
        recs.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return np.array(recs), np.array(precs)


def _classification_report(y_true, y_pred) -> str:
    lines = [
        "              precision    recall  f1-score   support",
        "",
    ]
    for label, name in [(0, "Legitimate"), (1, "Phishing")]:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        support = (y_true == label).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lines.append(
            f"  {name:<10} {prec:>10.4f} {rec:>9.4f} {f1:>9.4f} {support:>9d}"
        )
    acc = (y_true == y_pred).mean()
    support_total = len(y_true)
    lines.append("")
    lines.append(f"    accuracy {'':>20} {acc:>9.4f} {support_total:>9d}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def _save_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticklabels(["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Test Set (NumPy Demo)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_roc(fpr, tpr, auc_val, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set (NumPy Demo)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_pr(rec, prec, out_path):
    auc = np.trapz(prec[np.argsort(rec)], np.sort(rec))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR-AUC = {abs(auc):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Test Set (NumPy Demo)")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_feature_importance(weights, feature_names, out_path, top_k=20):
    importances = np.abs(weights)
    order = np.argsort(importances)[::-1][:top_k]
    top_features = np.array(feature_names)[order]
    top_values = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(top_features))[::-1]
    ax.barh(y_pos, top_values, color="steelblue")
    ax.set_yticks(y_pos); ax.set_yticklabels(top_features)
    ax.set_xlabel("|Standardised weight|")
    ax.set_title("Top 20 Feature Importances (NumPy Demo)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_class_distribution(y, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = np.bincount(y, minlength=2)
    ax.bar(["Legitimate", "Phishing"], counts, color=["seagreen", "firebrick"])
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v),
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Count"); ax.set_title("Class distribution (full dataset)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> dict:
    t_start = time.time()
    cfg = load_config()
    lbl_col = cfg["columns"]["label_column"]

    # 1. Load + clean + engineer features
    df = load_raw_dataset()
    df = clean_dataframe(df)
    df = engineer_features(df)

    # 2. Split
    train_df, val_df, test_df = split_data(df)

    feature_cols = get_feature_column_names()
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[lbl_col].to_numpy(dtype=int)
    X_val   = val_df[feature_cols].to_numpy(dtype=float)
    y_val   = val_df[lbl_col].to_numpy(dtype=int)
    X_test  = test_df[feature_cols].to_numpy(dtype=float)
    y_test  = test_df[lbl_col].to_numpy(dtype=int)

    # 3. Train
    logger.info("Training NumPy LogisticRegression…")
    model = NumpyLogisticRegression(
        lr=0.05, epochs=200, batch_size=256, l2=1e-3, random_state=42,
    )
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start
    logger.info("Training finished in %.2fs", fit_time)

    # 4. Evaluate
    results = {
        "train": _metrics(y_train, model.predict_proba(X_train)[:, 1], "train"),
        "val":   _metrics(y_val,   model.predict_proba(X_val)[:, 1],   "val"),
        "test":  _metrics(y_test,  model.predict_proba(X_test)[:, 1],  "test"),
        "training_time_seconds": round(fit_time, 3),
        "model_type":   "NumpyLogisticRegression (demo)",
    }

    # Simple CV-like repetition (no sklearn): 5 random shuffles.
    rng = np.random.default_rng(42)
    cv_f1 = []
    for _ in range(5):
        idx = rng.permutation(len(X_train))
        fold_size = len(idx) // 5
        val_idx = idx[:fold_size]
        trn_idx = idx[fold_size:]
        m = NumpyLogisticRegression(
            lr=0.05, epochs=100, batch_size=256, l2=1e-3,
            random_state=int(rng.integers(0, 10000)),
        )
        m.fit(X_train[trn_idx], y_train[trn_idx])
        cv_f1.append(_metrics(
            y_train[val_idx], m.predict_proba(X_train[val_idx])[:, 1], "cv"
        )["f1"])
    results["cv_f1_mean"] = float(np.mean(cv_f1))
    results["cv_f1_std"]  = float(np.std(cv_f1))

    # 5. Save artefacts
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])
    models_dir  = ensure_dir(project_root() / cfg["paths"]["models_dir"])

    with (results_dir / "metrics_summary.json").open("w") as fh:
        json.dump(results, fh, indent=2)

    y_pred_test = model.predict(X_test)
    cm = _confusion_matrix(y_test, y_pred_test)
    _save_confusion_matrix(cm, results_dir / "confusion_matrix.png")

    fpr, tpr = _roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    _save_roc(fpr, tpr, results["test"]["roc_auc"], results_dir / "roc_curve.png")

    rec, prec = _pr_curve(y_test, model.predict_proba(X_test)[:, 1])
    _save_pr(rec, prec, results_dir / "pr_curve.png")

    _save_feature_importance(model.W, feature_cols,
                             results_dir / "feature_importance.png")
    _save_class_distribution(df[lbl_col].to_numpy(),
                             results_dir / "class_distribution.png")

    with (results_dir / "classification_report.txt").open("w") as fh:
        fh.write(_classification_report(y_test, y_pred_test))

    # Save the model's weights + scaler so prediction can be reproduced.
    model_payload = {
        "model_type": "NumpyLogisticRegression",
        "feature_columns": feature_cols,
        "mean": model.mean_.tolist(),
        "std":  model.std_.tolist(),
        "weights": model.W.tolist(),
        "bias": float(model.b),
        "training_info": results,
    }
    with (models_dir / "numpy_demo_model.json").open("w") as fh:
        json.dump(model_payload, fh, indent=2)
    with (models_dir / "feature_columns.json").open("w") as fh:
        json.dump(feature_cols, fh, indent=2)

    logger.info("Pipeline completed in %.2fs", time.time() - t_start)
    return results


if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary, indent=2))

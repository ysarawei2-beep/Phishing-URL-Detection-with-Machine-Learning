"""
Factory for the single ML model used in this project.

Why Random Forest?
------------------
* It handles mixed-scale features without normalisation.
* It is robust to outliers and noisy features.
* It gives us a natural feature-importance readout, which is very
  useful for understanding which URL traits drive phishing
  predictions.
* It trains quickly even on laptops with CPU-only scikit-learn.

These are the reasons Hannousse & Yahiouche (2021) also chose
Random Forest as a strong baseline on URL datasets.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, load_config

logger = get_logger(__name__)


def build_model() -> Pipeline:
    """
    Build a scikit-learn ``Pipeline`` containing a StandardScaler
    (for interpretability of feature magnitudes) and a
    ``RandomForestClassifier`` configured from ``config.yaml``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline ready for ``.fit(X, y)``.
    """
    cfg = load_config()["model"]

    logger.info(
        "Creating Random Forest with n_estimators=%d, max_depth=%s",
        cfg["n_estimators"], cfg["max_depth"],
    )

    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight=cfg["class_weight"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
    )

    pipeline = Pipeline(
        steps=[
            # Scaler keeps feature magnitudes comparable – helpful
            # even for tree models when inspecting importances.
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipeline

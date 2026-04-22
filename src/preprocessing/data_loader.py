"""
Load the Kaggle *Phishing Dataset for Machine Learning* CSV and
produce reproducible train / validation / test splits.

Dataset reference
-----------------
Tan, Choon Lin (2018). *Phishing Dataset for Machine Learning:
Feature Evaluation*.  Mendeley Data / Kaggle (shashwatwork).
The CSV contains 10 000 balanced rows (5 000 phishing, 5 000
legitimate) with 48 pre-extracted features and a ``CLASS_LABEL``
target (1 = phishing, 0 = legitimate).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, project_root
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


def _resolve_dataset_path() -> Path:
    """Return the raw CSV path; fall back to the bundled sample."""
    cfg = load_config()
    raw_path = project_root() / cfg["paths"]["raw_data"]
    sample_path = project_root() / cfg["paths"]["sample_data"]

    if raw_path.exists():
        logger.info("Using Kaggle dataset at %s", raw_path)
        return raw_path

    logger.warning(
        "Full dataset not found at %s – falling back to the bundled "
        "sample at %s.  Download the full Kaggle dataset for final "
        "results.",
        raw_path, sample_path,
    )
    return sample_path


def load_raw_dataset() -> pd.DataFrame:
    """
    Load the raw dataset as a ``pandas.DataFrame``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the identifier column removed.
        ``CLASS_LABEL`` column contains the integer target.
    """
    cfg = load_config()
    id_col = cfg["columns"]["id_column"]
    lbl_col = cfg["columns"]["label_column"]

    path = _resolve_dataset_path()
    df = pd.read_csv(path)

    # Drop the ID column (it is not a feature).
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    if lbl_col not in df.columns:
        raise ValueError(
            f"Dataset at {path} does not contain the expected "
            f"label column '{lbl_col}'. Found columns: {list(df.columns)}"
        )

    # Ensure the label is integer 0/1.
    df[lbl_col] = df[lbl_col].astype(int)

    # Optional developer sub-sampling for fast experimentation.
    if cfg["runtime"]["sample_for_dev"]:
        n = min(cfg["runtime"]["dev_sample_size"], len(df))
        df = df.sample(n=n, random_state=cfg["split"]["random_state"])
        logger.info("Developer mode: reduced dataset to %d rows", n)

    logger.info(
        "Loaded %d rows and %d columns from %s",
        len(df), df.shape[1], path.name,
    )
    return df.reset_index(drop=True)


def _numpy_stratified_split(
    df: pd.DataFrame,
    lbl_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    do_stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified three-way split using only NumPy (sklearn fallback)."""
    rng = np.random.default_rng(random_state)

    if do_stratify:
        groups = [
            df[df[lbl_col] == cls].reset_index(drop=True)
            for cls in sorted(df[lbl_col].unique())
        ]
    else:
        groups = [df.reset_index(drop=True)]

    train_parts, val_parts, test_parts = [], [], []
    for g in groups:
        idx = rng.permutation(len(g))
        g = g.iloc[idx].reset_index(drop=True)
        n_test = int(round(len(g) * test_size))
        n_val = int(round(len(g) * val_size))
        test_parts.append(g.iloc[:n_test])
        val_parts.append(g.iloc[n_test:n_test + n_val])
        train_parts.append(g.iloc[n_test + n_val:])

    train = pd.concat(train_parts).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    val = pd.concat(val_parts).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    test = pd.concat(test_parts).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    return train, val, test


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 train / validation / test split.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_raw_dataset`.

    Returns
    -------
    (train, val, test) DataFrames
    """
    cfg = load_config()
    lbl_col = cfg["columns"]["label_column"]
    random_state = cfg["split"]["random_state"]
    test_size    = cfg["split"]["test_size"]
    val_size     = cfg["split"]["val_size"]
    do_stratify  = cfg["split"]["stratify"]

    # Try to use sklearn if it is available; otherwise fall back to a
    # pure-numpy stratified split so that the pipeline still runs in
    # locked-down environments (e.g. the demo runner).
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        stratify = df[lbl_col] if do_stratify else None
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        stratify_tv = train_val[lbl_col] if do_stratify else None
        relative_val_size = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_tv,
        )
    except ImportError:
        logger.warning(
            "scikit-learn is not installed – using the numpy fallback "
            "splitter."
        )
        train, val, test = _numpy_stratified_split(
            df, lbl_col, test_size, val_size, random_state, do_stratify,
        )

    logger.info(
        "Split sizes – train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )

    # Persist the splits so results are reproducible.
    out_dir = ensure_dir(project_root() / cfg["paths"]["processed_dir"])
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv",   index=False)
    test.to_csv(out_dir / "test.csv",  index=False)
    logger.info("Saved processed splits to %s", out_dir)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )

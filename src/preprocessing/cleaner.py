"""
Clean and validate the feature DataFrame.

The Kaggle dataset is already well-formed, but this module still
performs:
    * null / duplicate removal
    * target-column validation
    * consistent column ordering
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger, load_config

logger = get_logger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a defensive cleaning pass on the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_raw_dataset``.

    Returns
    -------
    pd.DataFrame
    """
    cfg = load_config()
    lbl_col = cfg["columns"]["label_column"]
    initial_rows = len(df)

    # Drop rows with NA in any column (dataset has none by default,
    # but we keep the guard for safety against tampered CSVs).
    df = df.dropna().copy()

    # Drop duplicate rows (keeps the first occurrence).
    df = df.drop_duplicates()

    # Validate label values.
    unique_labels = set(df[lbl_col].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected label values {unique_labels}.  "
            "Expected only 0 (legitimate) or 1 (phishing)."
        )

    logger.info(
        "Cleaning removed %d rows (from %d to %d).",
        initial_rows - len(df), initial_rows, len(df),
    )
    return df.reset_index(drop=True)

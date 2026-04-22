"""Unit tests for data loading & cleaning."""
import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_dataframe
from src.preprocessing.data_loader import load_raw_dataset, split_data


class TestCleanDataframe:
    def test_drops_nans(self):
        df = pd.DataFrame(
            {"UrlLength": [10, None, 30], "CLASS_LABEL": [0, 1, 1]}
        )
        cleaned = clean_dataframe(df)
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) == 2

    def test_drops_duplicates(self):
        df = pd.DataFrame(
            {"UrlLength": [10, 10, 30], "CLASS_LABEL": [0, 0, 1]}
        )
        cleaned = clean_dataframe(df)
        assert len(cleaned) == 2

    def test_invalid_labels_raise(self):
        df = pd.DataFrame(
            {"UrlLength": [10, 20], "CLASS_LABEL": [0, 2]}
        )
        with pytest.raises(ValueError):
            clean_dataframe(df)


class TestLoaderAndSplit:
    def test_load_raw_dataset_returns_dataframe(self):
        df = load_raw_dataset()
        assert isinstance(df, pd.DataFrame)
        assert "CLASS_LABEL" in df.columns
        assert len(df) > 0

    def test_split_sizes_sum_to_total(self):
        df = load_raw_dataset()
        train, val, test = split_data(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_split_preserves_label_balance(self):
        df = load_raw_dataset()
        train, val, test = split_data(df)
        total_pos = df["CLASS_LABEL"].mean()
        # Stratified split should give similar class proportions.
        for part in (train, val, test):
            assert abs(part["CLASS_LABEL"].mean() - total_pos) < 0.05

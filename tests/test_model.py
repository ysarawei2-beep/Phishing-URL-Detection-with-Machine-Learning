"""Unit tests for model construction and a tiny smoke-train."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import (
    engineer_features,
    get_feature_column_names,
)
from src.models import build_model


class TestBuildModel:
    def test_returns_pipeline_with_random_forest(self):
        model = build_model()
        # Final step name must be "clf" and be a RandomForestClassifier.
        name, clf = model.steps[-1]
        assert name == "clf"
        assert isinstance(clf, RandomForestClassifier)

    def test_has_scaler(self):
        model = build_model()
        assert "scaler" in dict(model.steps)


class TestSmokeTrain:
    """Fit the pipeline on a tiny synthetic dataset."""

    @pytest.fixture
    def tiny_dataset(self):
        rng = np.random.default_rng(0)
        cols = get_feature_column_names()
        # 200 random rows, 0/1 label
        X = pd.DataFrame(rng.integers(0, 100, size=(200, len(cols))), columns=cols)
        y = pd.Series(rng.integers(0, 2, size=200))
        return X, y

    def test_fit_and_predict(self, tiny_dataset):
        X, y = tiny_dataset
        model = build_model()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_between_zero_and_one(self, tiny_dataset):
        X, y = tiny_dataset
        model = build_model().fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestEngineerFeaturesIsSafe:
    def test_no_infinities_when_zero_urllength(self):
        from src.feature_engineering import KAGGLE_FEATURE_COLUMNS
        data = {c: [1] for c in KAGGLE_FEATURE_COLUMNS}
        data["UrlLength"] = [0]
        data["CLASS_LABEL"] = [0]
        df = pd.DataFrame(data)
        enriched = engineer_features(df)
        assert np.isfinite(enriched.to_numpy(dtype=float)).all()

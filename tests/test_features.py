"""Unit tests for the URL-level feature extractor."""
import pandas as pd
import pytest

from src.feature_engineering import (
    extract_features_from_url,
    extract_features_dataframe_from_urls,
    get_feature_column_names,
    engineer_features,
    KAGGLE_FEATURE_COLUMNS,
)


class TestExtractFeaturesFromUrl:
    def test_returns_all_expected_columns(self):
        feats = extract_features_from_url("https://www.example.com/path?x=1")
        expected = set(get_feature_column_names())
        assert set(feats.keys()) == expected

    def test_https_flag(self):
        secure = extract_features_from_url("https://example.com")
        insecure = extract_features_from_url("http://example.com")
        assert secure["NoHttps"] == 0
        assert insecure["NoHttps"] == 1

    def test_ip_detection(self):
        ip_url = extract_features_from_url("http://192.168.1.1/login")
        name_url = extract_features_from_url("http://example.com/login")
        assert ip_url["IpAddress"] == 1
        assert name_url["IpAddress"] == 0

    def test_at_symbol(self):
        feats = extract_features_from_url("http://example.com/login@bad.com")
        assert feats["AtSymbol"] == 1

    def test_sensitive_keywords(self):
        feats = extract_features_from_url("http://paypal.com/login/secure")
        # Two suspicious tokens: "login" and "secure"
        assert feats["NumSensitiveWords"] >= 2

    def test_empty_url_returns_zeroed_vector(self):
        feats = extract_features_from_url("")
        assert all(v == 0.0 for v in feats.values())

    def test_url_length_matches(self):
        url = "https://my-bank.com/account/verify"
        feats = extract_features_from_url(url)
        assert feats["UrlLength"] == len(url)

    def test_non_string_input(self):
        feats = extract_features_from_url(None)  # type: ignore[arg-type]
        assert all(v == 0.0 for v in feats.values())


class TestExtractFeaturesDataframe:
    def test_shape_and_columns(self):
        urls = pd.Series([
            "https://example.com",
            "http://phish.tk/login",
            "http://1.2.3.4/account",
        ])
        df = extract_features_dataframe_from_urls(urls)
        assert len(df) == 3
        assert list(df.columns) == get_feature_column_names()

    def test_no_nulls(self):
        urls = pd.Series(["http://a.com", "http://b.com/login?q=1"])
        df = extract_features_dataframe_from_urls(urls)
        assert df.isna().sum().sum() == 0


class TestEngineerFeatures:
    def _sample_df(self):
        # minimal valid DataFrame containing the features required
        # by engineer_features
        data = {c: [1, 2] for c in KAGGLE_FEATURE_COLUMNS}
        data["UrlLength"]      = [100, 50]
        data["PathLength"]     = [20, 10]
        data["HostnameLength"] = [30, 15]
        data["NumNumericChars"] = [5, 2]
        data["CLASS_LABEL"] = [0, 1]
        return pd.DataFrame(data)

    def test_engineered_columns_added(self):
        df = self._sample_df()
        enriched = engineer_features(df)
        for c in ("PathToUrlRatio", "HostnameToUrlRatio", "DigitToUrlRatio"):
            assert c in enriched.columns

    def test_ratios_are_correct(self):
        df = self._sample_df()
        enriched = engineer_features(df)
        # First row: 20/100 = 0.2, 30/100 = 0.3, 5/100 = 0.05
        assert enriched["PathToUrlRatio"].iloc[0]     == pytest.approx(0.2)
        assert enriched["HostnameToUrlRatio"].iloc[0] == pytest.approx(0.3)
        assert enriched["DigitToUrlRatio"].iloc[0]    == pytest.approx(0.05)

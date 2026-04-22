"""
Feature engineering for the Phishing URL Detection project.

This module has two responsibilities:

1. `engineer_features` – enrich the pre-extracted Kaggle dataset
   with a few domain-motivated *ratio* features that the original
   CSV does not contain.

2. `extract_features_from_url` – build an identical feature row
   from a raw URL string.  This is what the command-line predictor
   uses at inference time.  Because the Kaggle CSV contains features
   that require fetching the HTML (e.g. ``PctExtHyperlinks``), this
   lexical-only extractor fills those fields with sensible
   placeholders (0.0) and relies on the URL-level signals to make
   the prediction.  See README.md for the caveats.
"""
from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import tldextract  # type: ignore
    _TLDEXTRACT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep, pip install tldextract
    tldextract = None  # type: ignore
    _TLDEXTRACT_AVAILABLE = False

from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Named constants – avoid "magic" tokens in the rest of the codebase.
# ----------------------------------------------------------------------
# Words often abused in phishing URLs (Sahoo et al., 2017).
SUSPICIOUS_KEYWORDS: tuple = (
    "login", "secure", "account", "update", "bank", "verify",
    "ebay", "paypal", "webscr", "signin", "confirm", "wp-admin",
    "admin", "password", "billing",
)

# Regex for a bare IPv4 address in place of a hostname.
_IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

# The 48 feature columns from the Kaggle CSV (everything except
# ``id`` and ``CLASS_LABEL``).
KAGGLE_FEATURE_COLUMNS: List[str] = [
    "NumDots", "SubdomainLevel", "PathLevel", "UrlLength",
    "NumDash", "NumDashInHostname", "AtSymbol", "TildeSymbol",
    "NumUnderscore", "NumPercent", "NumQueryComponents",
    "NumAmpersand", "NumHash", "NumNumericChars", "NoHttps",
    "RandomString", "IpAddress", "DomainInSubdomains",
    "DomainInPaths", "HttpsInHostname", "HostnameLength",
    "PathLength", "QueryLength", "DoubleSlashInPath",
    "NumSensitiveWords", "EmbeddedBrandName", "PctExtHyperlinks",
    "PctExtResourceUrls", "ExtFavicon", "InsecureForms",
    "RelativeFormAction", "ExtFormAction", "AbnormalFormAction",
    "PctNullSelfRedirectHyperlinks", "FrequentDomainNameMismatch",
    "FakeLinkInStatusBar", "RightClickDisabled", "PopUpWindow",
    "SubmitInfoToEmail", "IframeOrFrame", "MissingTitle",
    "ImagesOnlyInForm", "SubdomainLevelRT", "UrlLengthRT",
    "PctExtResourceUrlsRT", "AbnormalExtFormActionR",
    "ExtMetaScriptLinkRT", "PctExtNullSelfRedirectHyperlinksRT",
]

# Engineered features we add on top of KAGGLE_FEATURE_COLUMNS.
ENGINEERED_FEATURE_COLUMNS: List[str] = [
    "PathToUrlRatio",
    "HostnameToUrlRatio",
    "DigitToUrlRatio",
]


# ----------------------------------------------------------------------
# Feature engineering on the Kaggle DataFrame
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a handful of ratio features to the DataFrame.

    Why ratios?  Absolute lengths can be misleading (a long CDN URL
    can still be legitimate).  Ratios normalise for overall URL
    length and often expose phishing patterns.
    """
    cfg = load_config()
    if not cfg["features"]["add_engineered"]:
        logger.info("Skipping engineered features (disabled in config).")
        return df.copy()

    df = df.copy()
    # Guard against division-by-zero: replace 0-length URLs with 1.
    safe_url_length = df["UrlLength"].replace(0, 1)

    df["PathToUrlRatio"] = df["PathLength"] / safe_url_length
    df["HostnameToUrlRatio"] = df["HostnameLength"] / safe_url_length
    df["DigitToUrlRatio"] = df["NumNumericChars"] / safe_url_length

    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    logger.info(
        "Added %d engineered features: %s",
        len(ENGINEERED_FEATURE_COLUMNS),
        ENGINEERED_FEATURE_COLUMNS,
    )
    return df


def get_feature_column_names() -> List[str]:
    """Return the final ordered list of feature column names."""
    cfg = load_config()
    if cfg["features"]["add_engineered"]:
        return KAGGLE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
    return list(KAGGLE_FEATURE_COLUMNS)


# ----------------------------------------------------------------------
# URL-only feature extractor (used by predict.py)
# ----------------------------------------------------------------------
def _safe_urlparse(url: str):
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


def _has_ip_address(hostname: str) -> int:
    return int(bool(_IP_PATTERN.match(hostname)))


def extract_features_from_url(url: str) -> Dict[str, float]:
    """
    Build a feature dict for a single raw URL.

    The CSV dataset includes HTML-level features that we cannot
    compute without fetching the page.  This extractor returns 0.0
    for those columns; the URL-only signals are enough to drive a
    reasonable prediction at the command line.

    Parameters
    ----------
    url : str

    Returns
    -------
    Dict[str, float]
        Feature names -> values, in the same order as
        :func:`get_feature_column_names`.
    """
    if not isinstance(url, str) or not url:
        return {name: 0.0 for name in get_feature_column_names()}

    parsed = _safe_urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    if _TLDEXTRACT_AVAILABLE:
        extracted = tldextract.extract(url)
        subdomain = extracted.subdomain
        domain = extracted.domain
    else:
        # Fallback if tldextract is not installed.  We do a best-effort
        # split: treat the last two labels of the hostname as the
        # registered domain and anything before that as the subdomain.
        labels = hostname.split(".") if hostname else []
        if len(labels) >= 2:
            subdomain = ".".join(labels[:-2])
            domain = labels[-2]
        else:
            subdomain = ""
            domain = labels[0] if labels else ""
    subdomain_parts = subdomain.split(".") if subdomain else []

    num_digits = sum(ch.isdigit() for ch in url)
    url_len = len(url) or 1

    features: Dict[str, float] = {name: 0.0 for name in KAGGLE_FEATURE_COLUMNS}

    # Lexical features we CAN compute from just the URL.
    features.update(
        {
            "NumDots":            url.count("."),
            "SubdomainLevel":     len(subdomain_parts),
            "PathLevel":          max(path.count("/") - 1, 0),
            "UrlLength":          len(url),
            "NumDash":            url.count("-"),
            "NumDashInHostname":  hostname.count("-"),
            "AtSymbol":           int("@" in url),
            "TildeSymbol":        int("~" in url),
            "NumUnderscore":      url.count("_"),
            "NumPercent":         url.count("%"),
            "NumQueryComponents": len(query.split("&")) if query else 0,
            "NumAmpersand":       url.count("&"),
            "NumHash":            url.count("#"),
            "NumNumericChars":    num_digits,
            "NoHttps":            int(parsed.scheme.lower() != "https"),
            "IpAddress":          _has_ip_address(hostname),
            "DomainInSubdomains": int(domain.lower() in subdomain.lower()) if domain else 0,
            "DomainInPaths":      int(domain.lower() in path.lower()) if domain else 0,
            "HttpsInHostname":    int("https" in hostname.lower()),
            "HostnameLength":     len(hostname),
            "PathLength":         len(path),
            "QueryLength":        len(query),
            "DoubleSlashInPath":  int("//" in path),
            "NumSensitiveWords":  sum(
                k in url.lower() for k in SUSPICIOUS_KEYWORDS
            ),
            # RT ("runtime") columns use the tri-state convention
            # 1 = safe, 0 = unknown, -1 = suspicious.  We default to 0.
            "SubdomainLevelRT":   0,
            "UrlLengthRT":        0,
        }
    )

    # Add engineered ratio features if enabled in config.
    cfg = load_config()
    if cfg["features"]["add_engineered"]:
        features["PathToUrlRatio"]      = len(path) / url_len
        features["HostnameToUrlRatio"]  = len(hostname) / url_len
        features["DigitToUrlRatio"]     = num_digits / url_len

    # Return in the canonical column order.
    ordered_names = get_feature_column_names()
    return {name: float(features.get(name, 0.0)) for name in ordered_names}


def extract_features_dataframe_from_urls(urls: pd.Series) -> pd.DataFrame:
    """Vectorised wrapper – used when scoring a list of URLs."""
    logger.info("Extracting URL features from %d inputs…", len(urls))
    records = [extract_features_from_url(u) for u in urls]
    return pd.DataFrame.from_records(records, columns=get_feature_column_names())

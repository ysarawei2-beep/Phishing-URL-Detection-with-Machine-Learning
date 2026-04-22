"""
Small generic helpers used across the codebase:

* `project_root()` – absolute path to the repository root.
* `load_config()`  – reads ``config/config.yaml`` once and returns
                     it as a nested dict.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


# ----------------------------------------------------------------------
# Named constants
# ----------------------------------------------------------------------
_CONFIG_RELATIVE_PATH = "config/config.yaml"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """
    Return the absolute path to the project root.

    The project root is considered the parent directory of ``src``.
    Using ``lru_cache`` guarantees the computation only happens once
    per process.
    """
    # src/utils/helpers.py  ->  src/utils  ->  src  ->  project root
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load ``config/config.yaml`` and cache the result."""
    cfg_path = project_root() / _CONFIG_RELATIVE_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find configuration file at {cfg_path}.  "
            "Make sure you run the scripts from the project root."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if it does not exist, return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

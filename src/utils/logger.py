"""
Simple logging helper.

A single call to `get_logger(__name__)` returns a ready-to-use
logger that prints to the console with a consistent format.
"""
import logging
import sys


# ----------------------------------------------------------------------
# Named constants – avoid "magic strings" in the rest of the codebase.
# ----------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Guard against adding duplicate handlers when the function
    # is called multiple times (e.g. inside Jupyter notebooks).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger

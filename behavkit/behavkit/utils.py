"""General utilities shared across the other modules."""
import logging
import numpy as np

logger = logging.getLogger("behavkit")


def setup_logging(level=logging.INFO):
    """Configure basic logging. Call this once at the start of your script/notebook."""
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    return logger


def to_seconds(timestamp_str, context=""):
    """
    Convert a 'HH:MM:SS' timestamp (common format in manual scoring
    spreadsheets) to seconds. Logs (instead of silently swallowing) any
    timestamp that fails to parse, so lost annotations don't disappear
    without a trace.
    """
    try:
        h, m, s = map(int, str(timestamp_str).split(':'))
        return h * 3600 + m * 60 + s
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}' ({context}): {e}")
        return np.nan

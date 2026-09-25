"""Logging setup for the Amazon ML Challenge 2026 project."""

import logging
import os
import sys
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "amazon_ml_2026",
    level: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
            If None, reads from LOG_LEVEL environment variable or defaults to 'INFO'.
        log_format: Format string for log messages.
        date_format: Date format string for timestamps.

    Returns:
        Configured logging.Logger instance.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger

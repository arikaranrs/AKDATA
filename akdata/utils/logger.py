"""Structured Logging Module for AKDATA.

This module provides a pre-configured logger for all modules in AKDATA,
following standard enterprise logging practices.
"""

import logging
import sys

# Default format for logs
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get or create a configured logger with the specified name.

    Args:
        name (str): The name of the module requesting the logger.

    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger doesn't have handlers configured yet, configure it
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Avoid propagation to root logger to prevent duplicate logs in some environments
        logger.propagate = False

    return logger

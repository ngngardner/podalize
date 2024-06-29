"""Logger for the Podalize app."""

import logging
import os

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the app."""
    logger = logging.getLogger(name)

    log_level = os.getenv("LOG_LEVEL", "INFO")
    level = logging.getLevelName(log_level)

    logger.setLevel(level)

    ch = RichHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger

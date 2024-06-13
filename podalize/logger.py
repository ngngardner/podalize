"""Logger for the Podalize app."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the app."""
    logger = logging.getLogger(name)

    log_level = os.getenv("LOG_LEVEL", "INFO")
    level = logging.getLevelName(log_level)

    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # noqa: COM812
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger

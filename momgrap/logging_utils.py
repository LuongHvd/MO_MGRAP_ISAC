"""Lightweight logging helpers for progress tracking during long runs.

``setup_logging`` configures a stdout handler (UTF-8, with timestamps) and an
optional log file.  Modules call ``get_logger()`` and emit ``logger.info(...)``;
nothing prints until logging is configured (done by ``run_experiments`` and
``experiments.run_all``).
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "momgrap"


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Configure the package logger to print to stdout (+ optional file)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def ensure_logging() -> logging.Logger:
    """Configure a default stdout logger if none has been set up yet."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        return setup_logging()
    return logger

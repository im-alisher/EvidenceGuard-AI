"""Consistent logging setup for library code, scripts, and experiments."""

from __future__ import annotations

import logging
import sys
import time
from typing import Final, TextIO

PACKAGE_LOGGER_NAME: Final = "evidenceguard"
_LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


def configure_logging(
    level: str | int = "INFO",
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger and return it.

    Repeated calls do not add duplicate handlers. Set ``force=True`` to replace
    handlers previously installed on the package logger.
    """
    resolved_level = _resolve_level(level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if not logger.handlers:
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        handler.setFormatter(_build_formatter())
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(resolved_level)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or one of its named children."""
    if not name or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def _resolve_level(level: str | int) -> int:
    if isinstance(level, bool):
        raise ValueError("Log level must be a name or an integer severity.")
    if isinstance(level, int):
        if level not in {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }:
            raise ValueError(f"Unsupported numeric log level: {level}")
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.strip().upper())
        if isinstance(resolved, int):
            return resolved
    raise ValueError(f"Unsupported log level: {level!r}")


def _build_formatter() -> logging.Formatter:
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    formatter.converter = time.gmtime
    return formatter

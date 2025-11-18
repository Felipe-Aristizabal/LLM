"""Basic logging configuration helpers.

This module defines a small helper to configure logging in a consistent
way across scripts, CLIs and applications.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Default logging format used by the project
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _env_log_level(default: int = logging.INFO) -> int:
    """Return a logging level from the TQ_LOG_LEVEL environment variable.

    The environment variable may contain names such as 'DEBUG', 'INFO',
    'WARNING', 'ERROR' or 'CRITICAL'. If it is not set or invalid, the
    provided default level is returned.
    """
    raw = os.getenv("TQ_LOG_LEVEL")
    if not raw:
        return default

    level = getattr(logging, raw.upper(), None)
    if isinstance(level, int):
        return level
    return default


def setup_logging(level: Optional[int] = None) -> None:
    """Configure the root logger with a common format and level.

    This function should be called once near the entrypoint of each CLI
    or application. If a level is not provided, the value is read from
    the TQ_LOG_LEVEL environment variable or falls back to INFO.
    """
    if level is None:
        level = _env_log_level()

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
    )

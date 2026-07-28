"""
Structured logging setup shared by every module.

Usage:
    from backend.core.logging import get_logger, setup_logging
    setup_logging(level="DEBUG")          # call once in CLI entry points
    log = get_logger(__name__)
    log.info("extracted features", extra={"state_id": state_id, "n": 14})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def _configure_root_logger(level: int = logging.INFO, logs_dir: str = "results/logs") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def setup_logging(level: str | int = "INFO", logs_dir: str = "results/logs") -> None:
    """Configure root logger.  Call once from CLI entry points.

    Parameters
    ----------
    level:
        Log level as a string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``) or an integer constant from the ``logging`` module.
    logs_dir:
        Directory for the ``pipeline.log`` file.
    """
    global _CONFIGURED
    _CONFIGURED = False  # allow re-configuration with a different level
    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    _configure_root_logger(level=numeric_level, logs_dir=logs_dir)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring handlers on first call."""
    _configure_root_logger()
    return logging.getLogger(name)

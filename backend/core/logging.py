"""
Structured logging setup shared by every module.

Usage:
    from backend.core.logging import get_logger
    log = get_logger(__name__)
    log.info("extracted features", extra={"state_id": state_id, "n": 14})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def _configure_root_logger(logs_dir: str = "results/logs") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring handlers on first call."""
    _configure_root_logger()
    return logging.getLogger(name)

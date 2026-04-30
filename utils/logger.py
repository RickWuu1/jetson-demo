"""
Structured logger with color output.
Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import sys
from pathlib import Path

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

_LOGGERS = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create a named logger (cached)."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        if _HAS_COLORLOG:
            handler = colorlog.StreamHandler(sys.stdout)
            handler.setFormatter(colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)s] %(name)s%(reset)s: %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG":    "cyan",
                    "INFO":     "green",
                    "WARNING":  "yellow",
                    "ERROR":    "red",
                    "CRITICAL": "bold_red",
                },
            ))
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            ))
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def add_file_handler(logger: logging.Logger, log_path: str):
    """Add a file handler to an existing logger."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger

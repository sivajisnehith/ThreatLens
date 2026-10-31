"""
ThreatLens Logging Utilities

Provides structured logging configuration with file rotation and console output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from threatlens.config import LOGGING_CONFIG


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (default: 'threatlens')
        level: Log level (default: from LOGGING_CONFIG)
        log_file: Log filename (default: from LOGGING_CONFIG)
        log_dir: Log directory (default: DATA_CONFIG.LOGS_DIR)

    Returns:
        Configured logger instance
    """
    from threatlens.config import DATA_CONFIG

    logger_name = name or "threatlens"
    log_level = level or LOGGING_CONFIG.LEVEL
    log_filename = log_file or LOGGING_CONFIG.LOG_FILE
    log_directory = log_dir or DATA_CONFIG.LOGS_DIR

    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / log_filename

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG.FORMAT,
        datefmt=LOGGING_CONFIG.DATE_FORMAT,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=LOGGING_CONFIG.MAX_BYTES,
        backupCount=LOGGING_CONFIG.BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(f"threatlens.{name}")


# Default logger instance
default_logger = setup_logging()


__all__ = ["setup_logging", "get_logger", "default_logger"]
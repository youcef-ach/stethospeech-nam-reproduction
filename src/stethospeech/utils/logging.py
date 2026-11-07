"""Structured logging setup for training, evaluation, and data processing."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "stethospeech", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a standardized logger.

    Args:
        name: Logger name.
        level: Logging level (default: logging.INFO).
        log_file: Optional file path to write log messages.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

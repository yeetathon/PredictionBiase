"""Logging configuration using loguru."""
import sys
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """Configure application logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    logger.add(
        "data/logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


# Initialize on import
import os
os.makedirs("data/logs", exist_ok=True)
setup_logging()

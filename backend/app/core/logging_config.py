"""Logging configuration for the application."""

import logging
import sys
from typing import Any, Dict

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging() -> None:
    """
    Configure application logging.
    
    Sets up structured logging with appropriate formatters and handlers.
    """
    # Configure root logger
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Set scarcity components to INFO
    logging.getLogger("scarcity").setLevel(logging.INFO)
    logging.getLogger("app").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

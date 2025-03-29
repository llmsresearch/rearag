import logging
import os
import sys
from typing import Optional


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up a logger with the given name and level.
    
    Args:
        name: The name of the logger
        level: The logging level (defaults to INFO if not provided)
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Get log level from environment variable or use default
    if level is None:
        env_level = os.environ.get("REARAG_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    
    logger.setLevel(level)
    
    # Create console handler if it doesn't exist yet
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger 
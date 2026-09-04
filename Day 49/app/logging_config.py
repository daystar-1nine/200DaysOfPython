# ==============================================================================
# Program    : Structured Logging Configuration Module (logging_config.py)
# Objective  : Configure application logger with ISO timestamps, severity levels, and request ID formatting.
# Concept    : Production Observability & Structured Logging
# Why Used   : Formats application log outputs for debugging without printing sensitive credentials.
# ==============================================================================

import logging
import sys
from app.config import settings

class StandardLogFormatter(logging.Formatter):
    """Custom Log Formatter incorporating timestamp, severity level, module, and message."""
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return super().format(record)

def setup_logging():
    """Initialize logging infrastructure across the application."""
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    log_format = "%(asctime)s | %(levelname)-7s | req_id=%(request_id)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = StandardLogFormatter(fmt=log_format, datefmt=date_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # Set third-party logger levels to WARN to minimize log clutter
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Retrieve logger instance for a given module name."""
    return logging.getLogger(name)

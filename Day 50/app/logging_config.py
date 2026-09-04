"""
===============================================================================
DAY 50 — STRUCTURED JSON LOGGING & MASKING SYSTEM
===============================================================================
This module configures structured JSON logging with ISO timestamps, log levels,
contextual request IDs, and sensitive field masking (passwords, JWT tokens).
===============================================================================
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


class SensitiveDataMaskerFormatter(logging.Formatter):
    """Custom logging formatter rendering JSON output and masking sensitive values."""

    SENSITIVE_KEYS = {"password", "token", "authorization", "secret", "card_number"}

    def _mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive keys in log payloads."""
        # What is used: Dictionary traversal with key sanitization.
        # Why it is used: Redacts sensitive parameters before writing to log streams.
        # How it works: Replaces string values of matching sensitive keys with '***MASKED***'.
        masked = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_KEYS:
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_dict(value)
            else:
                masked[key] = value
        return masked

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # What is used: Structured log payload dictionary construction.
        # Why it is used: Produces standardized JSON logs for log ingestion aggregators.
        # How it works: Extracts record metadata and serializes via json.dumps.
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "N/A"),
        }
        if isinstance(record.msg, dict):
            payload["data"] = self._mask_dict(record.msg)

        return json.dumps(payload)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Initialize root logger with JSON stream handler."""
    # What is used: Logger instantiation with StreamHandler.
    # Why it is used: Formats console standard output logs cleanly.
    # How it works: Sets level and attaches SensitiveDataMaskerFormatter.
    logger = logging.getLogger("taskflow")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(SensitiveDataMaskerFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logging()

"""
===============================================================================
DAY 50 — STANDARDIZED ERROR SCHEMAS
===============================================================================
This module defines standardized JSON error payload schemas returned by the API.
===============================================================================
"""

from typing import Any, Dict
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Detailed error object containing code, message, request_id, and details."""

    code: str
    message: str
    request_id: str
    details: Dict[str, Any] | None = None


class ErrorPayload(BaseModel):
    """Top-level error response envelope wrapping ErrorDetail."""

    error: ErrorDetail

"""
===============================================================================
DAY 50 — CUSTOM DOMAIN EXCEPTION TAXONOMY
===============================================================================
This module defines domain-specific custom exceptions mapped to HTTP status codes,
enabling clean error propagation to global error handling middleware.
===============================================================================
"""

from typing import Any, Dict


class TaskFlowException(Exception):
    """Base domain exception for TaskFlow API."""

    def __init__(self, message: str, status_code: int = 400, code: str = "BAD_REQUEST", details: Dict[str, Any] | None = None) -> None:
        # What is used: Base exception class initialization.
        # Why it is used: Encapsulates human message, HTTP status code, and machine error code.
        # How it works: Passes message to super() and stores attributes.
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


class AuthenticationError(TaskFlowException):
    """Exception raised when authentication fails (HTTP 401)."""

    def __init__(self, message: str = "Authentication credentials invalid or missing.") -> None:
        super().__init__(message=message, status_code=401, code="UNAUTHORIZED")


class ForbiddenError(TaskFlowException):
    """Exception raised when authorization is denied (HTTP 403)."""

    def __init__(self, message: str = "Operation forbidden. Insufficient permissions.") -> None:
        super().__init__(message=message, status_code=403, code="FORBIDDEN")


class NotFoundError(TaskFlowException):
    """Exception raised when a requested resource does not exist (HTTP 404)."""

    def __init__(self, message: str = "Requested resource not found.") -> None:
        super().__init__(message=message, status_code=404, code="NOT_FOUND")


class ConflictError(TaskFlowException):
    """Exception raised on duplicate constraint collisions (HTTP 409)."""

    def __init__(self, message: str = "Resource conflict detected.") -> None:
        super().__init__(message=message, status_code=409, code="CONFLICT")


class CustomValidationError(TaskFlowException):
    """Exception raised on validation rules violation (HTTP 422)."""

    def __init__(self, message: str = "Validation failed.", details: Dict[str, Any] | None = None) -> None:
        super().__init__(message=message, status_code=422, code="UNPROCESSABLE_ENTITY", details=details)

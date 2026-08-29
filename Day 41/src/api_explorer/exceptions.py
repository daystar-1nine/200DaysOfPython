# ==============================================================================
# Program    : API Exceptions Taxonomy (exceptions.py)
# Objective  : Custom API exception hierarchy (APIError, APIConnectionError, APINotFoundError, APIValidationError, APITimeoutError).
# Concept    : Advanced Exception Hierarchy (Day 37 requirement)
# Why Used   : Provides structured error domain taxonomy for network API errors.
# ==============================================================================

class APIError(Exception):
    """Base exception class for all REST API Explorer errors."""
    def __init__(self, message: str, code: str = "API_ERROR", status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code

    def __str__(self) -> str:
        status_str = f" [HTTP {self.status_code}]" if self.status_code else ""
        return f"[{self.code}]{status_str} {self.message}"

class APIConnectionError(APIError):
    """Raised when TCP/IP network connection to API host fails."""
    def __init__(self, message: str):
        super().__init__(message, code="CONNECTION_ERROR")

class APITimeoutError(APIError):
    """Raised when HTTP request exceeds timeout threshold."""
    def __init__(self, message: str):
        super().__init__(message, code="TIMEOUT_ERROR")

class APINotFoundError(APIError):
    """Raised when HTTP 404 Not Found is returned by server."""
    def __init__(self, message: str):
        super().__init__(message, code="NOT_FOUND_ERROR", status_code=404)

class APIValidationError(APIError):
    """Raised when client payload parameters fail validation."""
    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR", status_code=422)

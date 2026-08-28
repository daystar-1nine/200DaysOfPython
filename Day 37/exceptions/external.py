# ==============================================================================
# Program    : External Service Exceptions (external.py)
# Objective  : Define ExternalServiceError, TimeoutError, and AuthenticationError subclasses.
# Concept    : External API & Integration Exceptions
# Why Used   : Encapsulates network HTTP request and security authentication failures.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from base import ApplicationError

class ExternalServiceError(ApplicationError):
    """Raised when external HTTP API or microservice call fails."""
    def __init__(self, message: str):
        super().__init__(message, code="EXTERNAL_SERVICE_ERROR")

class TimeoutError(ExternalServiceError):
    """Raised when network request times out."""
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "TIMEOUT_ERROR"

class AuthenticationError(ApplicationError):
    """Raised when user credentials or auth tokens are invalid."""
    def __init__(self, message: str):
        super().__init__(message, code="AUTHENTICATION_ERROR")

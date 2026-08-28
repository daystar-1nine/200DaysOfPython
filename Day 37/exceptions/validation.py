# ==============================================================================
# Program    : Validation Exceptions (validation.py)
# Objective  : Define ValidationError, BoundsError, and FormatError subclasses.
# Concept    : Business Input Validation Exceptions
# Why Used   : Raised when user parameters fail business rules.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from base import ApplicationError

class ValidationError(ApplicationError):
    """Raised when input parameter validation fails."""
    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")

class BoundsError(ValidationError):
    """Raised when numerical values fall out of permitted bounds."""
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "BOUNDS_ERROR"

class FormatError(ValidationError):
    """Raised when text formatting fails (e.g. invalid email format)."""
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "FORMAT_ERROR"

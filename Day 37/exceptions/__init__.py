"""Day 37 Custom Exceptions Package."""
import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from base import ApplicationError
from validation import ValidationError, BoundsError, FormatError
from database import DatabaseError, UniqueConstraintError, NotFoundError
from external import ExternalServiceError, TimeoutError, AuthenticationError

__all__ = [
    "ApplicationError",
    "ValidationError",
    "BoundsError",
    "FormatError",
    "DatabaseError",
    "UniqueConstraintError",
    "NotFoundError",
    "ExternalServiceError",
    "TimeoutError",
    "AuthenticationError"
]

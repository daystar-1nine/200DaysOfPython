# ==============================================================================
# Program    : Database Exceptions (database.py)
# Objective  : Define DatabaseError, UniqueConstraintError, and NotFoundError subclasses.
# Concept    : Persistence Layer Exceptions
# Why Used   : Encapsulates relational database failures.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from base import ApplicationError

class DatabaseError(ApplicationError):
    """Raised when database query or connection execution fails."""
    def __init__(self, message: str):
        super().__init__(message, code="DATABASE_ERROR")

class UniqueConstraintError(DatabaseError):
    """Raised when duplicate unique constraint is violated."""
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "UNIQUE_CONSTRAINT_ERROR"

class NotFoundError(ApplicationError):
    """Raised when requested entity is not found in database."""
    def __init__(self, message: str):
        super().__init__(message, code="NOT_FOUND_ERROR")

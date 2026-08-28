# ==============================================================================
# Program    : PyFinance Custom Exception Hierarchy
# Objective  : Define custom exceptions for business, validation, repository, and API errors.
# Concept    : Custom Exception Hierarchy & Domain Errors
# Why Used   : Provides clean error messaging instead of unhandled tracebacks.
# ==============================================================================

class PyFinanceError(Exception):
    """Base exception for all PyFinance application errors."""
    pass

class ValidationError(PyFinanceError):
    """Raised when user input fails business validation rules."""
    pass

class NotFoundError(PyFinanceError):
    """Raised when requested entity is not found in database."""
    pass

class DatabaseError(PyFinanceError):
    """Raised when database query execution fails."""
    pass

class APIError(PyFinanceError):
    """Raised when external HTTP API request fails."""
    pass

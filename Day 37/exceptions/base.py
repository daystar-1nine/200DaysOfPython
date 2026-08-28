# ==============================================================================
# Program    : Base Application Exception (base.py)
# Objective  : Define root ApplicationError class with message and code attributes.
# Concept    : Exception Hierarchy Root
# Why Used   : Allows catching all application domain errors with single handler.
# ==============================================================================

class ApplicationError(Exception):
    """Base exception class for all domain application errors."""
    def __init__(self, message: str, code: str = "APPLICATION_ERROR"):
        # What is used : Custom Exception Attributes (message, code)
        # Why it is used: Provides structured error details for CLI presentation and logs
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

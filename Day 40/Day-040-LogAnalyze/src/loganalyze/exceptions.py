# ==============================================================================
# Program    : Application Custom Exceptions (exceptions.py)
# Objective  : Define LogAnalyze custom exception hierarchy (Day 37 requirement).
# Concept    : Exception Taxonomy & Chaining
# Why Used   : Provides structured error domain taxonomy for log analysis errors.
# ==============================================================================

class LogAnalyzeError(Exception):
    """Base exception class for all LogAnalyze application errors."""
    def __init__(self, message: str, code: str = "LOG_ANALYZE_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code

class InvalidLogError(LogAnalyzeError):
    """Raised when a log line or file format is invalid/malformed."""
    def __init__(self, message: str):
        super().__init__(message, code="INVALID_LOG_ERROR")

class FileProcessingError(LogAnalyzeError):
    """Raised when log file reading, opening, or streaming fails."""
    def __init__(self, message: str):
        super().__init__(message, code="FILE_PROCESSING_ERROR")

class ReportGenerationError(LogAnalyzeError):
    """Raised when report generation or export formatting fails."""
    def __init__(self, message: str):
        super().__init__(message, code="REPORT_GENERATION_ERROR")

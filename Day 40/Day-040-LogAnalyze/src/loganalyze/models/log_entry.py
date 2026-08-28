# ==============================================================================
# Program    : Log Level Enum & LogEntry Dataclass (log_entry.py)
# Objective  : Model LogLevel Enum and LogEntry value object (Day 39 requirement).
# Concept    : Dataclasses & Type-Safe Enumerations
# Why Used   : Provides immutable structure for single log line records.
# ==============================================================================

from dataclasses import dataclass
from enum import Enum

class LogLevel(Enum):
    """Enumeration of standard log levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass(frozen=True, slots=True)
class LogEntry:
    """Immutable LogEntry dataclass value object."""
    # What is used : @dataclass(frozen=True, slots=True)
    # Why it is used: Memory-efficient immutable representation of log lines
    timestamp: str
    level: LogLevel
    message: str

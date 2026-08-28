# ==============================================================================
# Program    : Log Parser, Generator & Iterator (log_parser.py)
# Objective  : Stream log files using generators (Day 35), context managers (Day 36), and custom iterators (Day 34).
# Concept    : Low-Memory Streaming Parser & Custom Iteration
# Why Used   : Parses gigabyte-sized log files with O(1) constant RAM footprint.
# ==============================================================================

from contextlib import contextmanager
import os
import re
import sys
from typing import Generator, Iterator, TextIO

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.exceptions import FileProcessingError, InvalidLogError
from loganalyze.models.log_entry import LogEntry, LogLevel

# Regex pattern matching: YYYY-MM-DD HH:MM:SS LEVEL Message
LOG_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(INFO|WARNING|ERROR)\s+(.*)$")

# What is used : Generator Function with yield (Day 35 requirement)
# Why it is used: Streams log entries lazily line-by-line without loading entire file into memory
def parse_log_line(line: str) -> LogEntry | None:
    """Parses a single log line into a LogEntry or returns None if empty/invalid."""
    cleaned = line.strip()
    if not cleaned:
        return None

    match = LOG_PATTERN.match(cleaned)
    if not match:
        return None

    timestamp, level_str, message = match.groups()
    try:
        level = LogLevel[level_str]
    except KeyError:
        return None

    return LogEntry(timestamp=timestamp, level=level, message=message)

# What is used : Context Manager (@contextmanager from contextlib) (Day 36 requirement)
# Why it is used: Guarantees safe file handle opening and cleanup
@contextmanager
def open_log_stream(filepath: str) -> Generator[TextIO, None, None]:
    """Context manager for safely opening and reading log files."""
    if not os.path.exists(filepath):
        raise FileProcessingError(f"Log file '{filepath}' does not exist.")

    file_handle = None
    try:
        file_handle = open(filepath, "r", encoding="utf-8", errors="ignore")
        yield file_handle
    except Exception as e:
        if not isinstance(e, FileProcessingError):
            raise FileProcessingError(f"Error opening log file '{filepath}': {e}") from e
        raise e
    finally:
        if file_handle and not file_handle.closed:
            file_handle.close()

def parse_log_stream(filepath: str) -> Generator[LogEntry, None, None]:
    """Generator streaming LogEntry items lazily from file path."""
    with open_log_stream(filepath) as stream:
        for line in stream:
            entry = parse_log_line(line)
            if entry is not None:
                yield entry

# What is used : Custom Iterator Protocol Class (__iter__, __next__) (Day 34 requirement)
# Why it is used: Implements stateful Iterator wrapping log entry sequences
class LogEntryIterator(Iterator[LogEntry]):
    """Custom Iterator iterating over a pre-parsed list of LogEntry records."""
    def __init__(self, entries: list[LogEntry]):
        self.entries = entries
        self._index = 0

    def __iter__(self) -> Iterator[LogEntry]:
        return self

    def __next__(self) -> LogEntry:
        if self._index >= len(self.entries):
            raise StopIteration
        entry = self.entries[self._index]
        self._index += 1
        return entry

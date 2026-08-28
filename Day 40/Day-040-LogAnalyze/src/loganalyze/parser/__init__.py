"""Log Parser Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.parser.log_parser import LogEntryIterator, parse_log_line, parse_log_stream, open_log_stream

__all__ = [
    "LogEntryIterator",
    "parse_log_line",
    "parse_log_stream",
    "open_log_stream"
]

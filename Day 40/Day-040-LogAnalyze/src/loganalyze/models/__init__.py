"""LogAnalyze Models Package."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.models.log_entry import LogLevel, LogEntry
from loganalyze.models.report import LogReport

__all__ = ["LogLevel", "LogEntry", "LogReport"]

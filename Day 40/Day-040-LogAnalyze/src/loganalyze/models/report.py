# ==============================================================================
# Program    : Log Report Model with Dunder Methods (report.py)
# Objective  : Model LogReport containing counts, errors, and dunder protocols (Day 32 requirement).
# Concept    : Dunder Method Protocols (__len__, __str__, __getitem__, __repr__)
# Why Used   : Enables native Python syntax on report instances (len(report), report[0], print(report)).
# ==============================================================================

from dataclasses import dataclass, field
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.models.log_entry import LogEntry, LogLevel

@dataclass
class LogReport:
    """Dataclass holding log analysis metrics and implementing dunder protocols."""
    total_lines: int = 0
    counts: dict[LogLevel, int] = field(default_factory=lambda: {
        LogLevel.INFO: 0,
        LogLevel.WARNING: 0,
        LogLevel.ERROR: 0
    })
    error_entries: list[LogEntry] = field(default_factory=list)

    # What is used : __len__ dunder method
    # Why it is used: Allows len(report) to return total analyzed log lines count
    def __len__(self) -> int:
        return self.total_lines

    # What is used : __getitem__ dunder method
    # Why it is used: Allows report[index] subscripting over error entries
    def __getitem__(self, index: int) -> LogEntry:
        return self.error_entries[index]

    # What is used : __repr__ dunder method
    # Why it is used: Returns developer-friendly string representation for debugging
    def __repr__(self) -> str:
        return f"<LogReport total_lines={self.total_lines} errors={len(self.error_entries)}>"

    # What is used : __str__ dunder method
    # Why it is used: Formats clean ASCII summary table for CLI output presentation
    def __str__(self) -> str:
        info_c = self.counts.get(LogLevel.INFO, 0)
        warn_c = self.counts.get(LogLevel.WARNING, 0)
        err_c = self.counts.get(LogLevel.ERROR, 0)
        return (
            f"Total Lines : {self.total_lines}\n\n"
            f"INFO        : {info_c}\n"
            f"WARNING     : {warn_c}\n"
            f"ERROR       : {err_c}"
        )

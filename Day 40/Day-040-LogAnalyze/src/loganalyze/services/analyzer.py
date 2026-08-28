# ==============================================================================
# Program    : Log Analyzer Service (analyzer.py)
# Objective  : Perform log analysis calculations using functional Python tools (Day 38) and @timer (Day 33).
# Concept    : Functional Log Transformation & Aggregation Pipeline
# Why Used   : Calculates log level counts, filters error entries, and performs keyword searches.
# ==============================================================================

from collections import Counter
import os
import sys
from typing import List

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.models.log_entry import LogEntry, LogLevel
from loganalyze.models.report import LogReport
from loganalyze.parser.log_parser import parse_log_stream
from loganalyze.utils.decorators import timer

class LogAnalyzer:
    """Core analysis engine for server log streams."""

    @timer
    def analyze_summary(self, filepath: str) -> LogReport:
        """Analyzes log file and returns LogReport containing level counts."""
        report = LogReport()
        # What is used : Streaming Generator Loop (Day 35 requirement)
        # Why it is used: Consumes generator yields line-by-line in O(1) RAM
        for entry in parse_log_stream(filepath):
            report.total_lines += 1
            report.counts[entry.level] = report.counts.get(entry.level, 0) + 1
            if entry.level == LogLevel.ERROR:
                report.error_entries.append(entry)
        return report

    @timer
    def get_errors(self, filepath: str) -> List[LogEntry]:
        """Returns list of ERROR log entries using filter() and lambda (Day 38 requirement)."""
        stream = parse_log_stream(filepath)
        # What is used : Functional filter() with lambda predicate
        # Why it is used: Filters log stream for ERROR level entries
        error_stream = filter(lambda e: e.level == LogLevel.ERROR, stream)
        return list(error_stream)

    @timer
    def search_logs(self, filepath: str, keyword: str) -> List[LogEntry]:
        """Searches log entries containing case-insensitive keyword via filter()."""
        keyword_lower = keyword.strip().lower()
        stream = parse_log_stream(filepath)
        search_stream = filter(lambda e: keyword_lower in e.message.lower(), stream)
        return list(search_stream)

    @timer
    def filter_by_date(self, filepath: str, target_date: str) -> List[LogEntry]:
        """Filters log entries matching specific YYYY-MM-DD date prefix."""
        stream = parse_log_stream(filepath)
        date_stream = filter(lambda e: e.timestamp.startswith(target_date), stream)
        return list(date_stream)

    def get_top_error_messages(self, filepath: str, top_n: int = 5) -> List[tuple[str, int]]:
        """Calculates top recurring error messages using Counter."""
        errors = self.get_errors(filepath)
        msg_counts = Counter(e.message for e in errors)
        return msg_counts.most_common(top_n)

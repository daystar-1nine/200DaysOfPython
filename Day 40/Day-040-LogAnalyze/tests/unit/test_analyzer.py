# ==============================================================================
# Test Suite : Log Analyzer Unit Tests (test_analyzer.py)
# Objective  : Test summary analysis, level counts, search, date filters, and timer profiling.
# Concept    : Unit Testing Log Analyzer Engine
# Why Used   : Asserts log metrics calculations and functional filtering pipelines.
# ==============================================================================

import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.models.log_entry import LogLevel
from loganalyze.services.analyzer import LogAnalyzer

def test_analyzer_summary_counts(tmp_path):
    f = tmp_path / "app.log"
    f.write_text(
        "2026-08-30 10:00:00 INFO Started\n"
        "2026-08-30 10:01:00 WARNING High RAM\n"
        "2026-08-30 10:02:00 ERROR DB failed\n"
        "2026-08-30 10:03:00 ERROR Timeout\n"
    )
    analyzer = LogAnalyzer()
    report = analyzer.analyze_summary(str(f))
    assert report.total_lines == 4
    assert report.counts[LogLevel.INFO] == 1
    assert report.counts[LogLevel.WARNING] == 1
    assert report.counts[LogLevel.ERROR] == 2
    assert len(report.error_entries) == 2

def test_analyzer_empty_log_file(tmp_path):
    f = tmp_path / "empty.log"
    f.write_text("")
    analyzer = LogAnalyzer()
    report = analyzer.analyze_summary(str(f))
    assert report.total_lines == 0
    assert report.counts[LogLevel.ERROR] == 0

def test_analyzer_get_errors(tmp_path):
    f = tmp_path / "app.log"
    f.write_text(
        "2026-08-30 10:00:00 INFO Started\n"
        "2026-08-30 10:02:00 ERROR DB failed\n"
    )
    analyzer = LogAnalyzer()
    errors = analyzer.get_errors(str(f))
    assert len(errors) == 1
    assert errors[0].message == "DB failed"

def test_analyzer_search_logs(tmp_path):
    f = tmp_path / "app.log"
    f.write_text(
        "2026-08-30 10:00:00 INFO Database connected\n"
        "2026-08-30 10:02:00 ERROR Database failed\n"
        "2026-08-30 10:03:00 INFO User login\n"
    )
    analyzer = LogAnalyzer()
    matches = analyzer.search_logs(str(f), "database")
    assert len(matches) == 2

def test_analyzer_filter_by_date(tmp_path):
    f = tmp_path / "app.log"
    f.write_text(
        "2026-08-30 10:00:00 INFO Day 1\n"
        "2026-08-31 10:00:00 INFO Day 2\n"
    )
    analyzer = LogAnalyzer()
    day1 = analyzer.filter_by_date(str(f), "2026-08-30")
    assert len(day1) == 1
    assert day1[0].message == "Day 1"

def test_top_error_messages(tmp_path):
    f = tmp_path / "app.log"
    f.write_text(
        "2026-08-30 10:00:00 ERROR DB Connection Failed\n"
        "2026-08-30 10:01:00 ERROR DB Connection Failed\n"
        "2026-08-30 10:02:00 ERROR Disk Full\n"
    )
    analyzer = LogAnalyzer()
    top = analyzer.get_top_error_messages(str(f), top_n=2)
    assert top[0] == ("DB Connection Failed", 2)
    assert top[1] == ("Disk Full", 1)

def test_timer_decorator_metadata(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("2026-08-30 10:00:00 INFO Test\n")
    analyzer = LogAnalyzer()
    analyzer.analyze_summary(str(f))
    assert hasattr(analyzer.analyze_summary, "last_execution_time")

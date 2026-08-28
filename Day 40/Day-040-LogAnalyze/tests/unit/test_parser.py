# ==============================================================================
# Test Suite : Log Parser Unit Tests (test_parser.py)
# Objective  : Test log line regex parsing, stream generators, context managers, and custom iterators.
# Concept    : Unit Testing Parser & Iterator Protocols
# Why Used   : Asserts log parsing correctness and exception raising for missing files.
# ==============================================================================

import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.exceptions import FileProcessingError
from loganalyze.models.log_entry import LogEntry, LogLevel
from loganalyze.parser.log_parser import (
    LogEntryIterator,
    parse_log_line,
    parse_log_stream,
    open_log_stream
)

def test_parse_valid_log_line():
    line = "2026-08-30 10:01:22 INFO User logged in"
    entry = parse_log_line(line)
    assert entry is not None
    assert entry.timestamp == "2026-08-30 10:01:22"
    assert entry.level == LogLevel.INFO
    assert entry.message == "User logged in"

def test_parse_empty_line_returns_none():
    assert parse_log_line("") is None
    assert parse_log_line("   \n") is None

def test_parse_invalid_format_returns_none():
    assert parse_log_line("Invalid Log Line Without Timestamp") is None

def test_parse_unknown_level_returns_none():
    assert parse_log_line("2026-08-30 10:01:22 DEBUG Unknown debug level") is None

def test_open_log_stream_missing_file_raises_error():
    with pytest.raises(FileProcessingError, match="does not exist"):
        with open_log_stream("non_existent_file_999.log"):
            pass

def test_open_log_stream_valid_file(tmp_path):
    f = tmp_path / "test.log"
    f.write_text("2026-08-30 10:01:22 INFO Hello\n")
    with open_log_stream(str(f)) as stream:
        content = stream.read()
        assert "2026-08-30" in content

def test_parse_log_stream_generator(tmp_path):
    f = tmp_path / "test_stream.log"
    f.write_text("2026-08-30 10:01:22 INFO Line 1\n2026-08-30 10:02:00 ERROR Line 2\n")
    entries = list(parse_log_stream(str(f)))
    assert len(entries) == 2
    assert entries[0].level == LogLevel.INFO
    assert entries[1].level == LogLevel.ERROR

def test_log_entry_iterator_custom_protocol():
    entries = [
        LogEntry("2026-08-30 10:00:00", LogLevel.INFO, "Msg 1"),
        LogEntry("2026-08-30 10:01:00", LogLevel.ERROR, "Msg 2")
    ]
    it = LogEntryIterator(entries)
    assert next(it).message == "Msg 1"
    assert next(it).message == "Msg 2"
    with pytest.raises(StopIteration):
        next(it)

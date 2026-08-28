# ==============================================================================
# Test Suite : CLI Integration Tests (test_cli.py)
# Objective  : Test loganalyze CLI subcommands (summary, errors, search, date, export).
# Concept    : Integration Testing CLI Commands
# Why Used   : Asserts end-to-end command-line dispatching and output formatting.
# ==============================================================================

import os
import sys
import unittest
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.main import main
from loganalyze.cli.commands import parse_and_execute

@pytest.fixture
def sample_log(tmp_path):
    p = tmp_path / "app.log"
    p.write_text(
        "2026-08-30 10:01:22 INFO User logged in\n"
        "2026-08-30 10:02:12 ERROR Database connection failed\n"
        "2026-08-30 10:02:15 WARNING High memory usage\n"
        "2026-08-31 11:00:00 INFO Next day event\n"
    )
    return str(p)

def test_cli_summary_subcommand(sample_log, capsys):
    parse_and_execute(["summary", sample_log])
    captured = capsys.readouterr().out
    assert "Total Lines : 4" in captured
    assert "INFO        : 2" in captured
    assert "ERROR       : 1" in captured

def test_cli_errors_subcommand(sample_log, capsys):
    parse_and_execute(["errors", sample_log])
    captured = capsys.readouterr().out
    assert "Found 1 Error Entries" in captured
    assert "Database connection failed" in captured

def test_cli_search_subcommand(sample_log, capsys):
    parse_and_execute(["search", sample_log, "memory"])
    captured = capsys.readouterr().out
    assert "Found 1 Matches for 'memory'" in captured
    assert "High memory usage" in captured

def test_cli_date_subcommand(sample_log, capsys):
    parse_and_execute(["date", sample_log, "2026-08-30"])
    captured = capsys.readouterr().out
    assert "Found 3 Entries for Date '2026-08-30'" in captured

def test_cli_export_subcommand(sample_log, tmp_path, capsys):
    out_json = str(tmp_path / "out_report.json")
    parse_and_execute(["export", sample_log, out_json])
    captured = capsys.readouterr().out
    assert "successfully exported to" in captured
    assert os.path.exists(out_json)

def test_cli_main_driver_success(sample_log):
    code = main(["summary", sample_log])
    assert code == 0

def test_cli_main_driver_file_error():
    code = main(["summary", "non_existent_file_999.log"])
    assert code == 1

class TestCLIRunner(unittest.TestCase):
    def test_cli_standalone(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()

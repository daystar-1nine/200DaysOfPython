# ==============================================================================
# Test Suite : Log Report & Exporter Unit Tests (test_reports.py)
# Objective  : Test LogReport dunder protocols (__len__, __str__, __getitem__) and JSON export.
# Concept    : Unit Testing Dunder Protocols & Report Serialization
# Why Used   : Asserts report model dunder behaviors and protocol export compliance.
# ==============================================================================

from dataclasses import FrozenInstanceError
import json
import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.exceptions import ReportGenerationError
from loganalyze.models.log_entry import LogEntry, LogLevel
from loganalyze.models.report import LogReport
from loganalyze.services.report_service import (
    JSONReportExporter,
    LogExporter,
    ReportService
)

def test_log_report_dunder_len():
    report = LogReport(total_lines=150)
    assert len(report) == 150

def test_log_report_dunder_getitem():
    e1 = LogEntry("2026-08-30 10:00:00", LogLevel.ERROR, "Err 1")
    e2 = LogEntry("2026-08-30 10:01:00", LogLevel.ERROR, "Err 2")
    report = LogReport(error_entries=[e1, e2])
    assert report[0] == e1
    assert report[1] == e2

def test_log_report_dunder_repr():
    report = LogReport(total_lines=50)
    assert "<LogReport total_lines=50" in repr(report)

def test_log_report_dunder_str_ascii():
    report = LogReport(
        total_lines=10,
        counts={LogLevel.INFO: 5, LogLevel.WARNING: 3, LogLevel.ERROR: 2}
    )
    s = str(report)
    assert "Total Lines : 10" in s
    assert "INFO        : 5" in s
    assert "ERROR       : 2" in s

def test_log_exporter_protocol_check():
    exporter = JSONReportExporter()
    assert isinstance(exporter, LogExporter)

def test_json_report_exporter(tmp_path):
    out = tmp_path / "report.json"
    e = LogEntry("2026-08-30 10:00:00", LogLevel.ERROR, "Test Error")
    report = LogReport(total_lines=1, counts={LogLevel.ERROR: 1}, error_entries=[e])

    exporter = JSONReportExporter()
    exporter.export(report, str(out))

    assert os.path.exists(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_lines"] == 1
    assert data["counts"]["ERROR"] == 1
    assert data["errors"][0]["message"] == "Test Error"

def test_report_service_export(tmp_path):
    out = tmp_path / "srv_report.json"
    report = LogReport(total_lines=5)
    srv = ReportService()
    srv.export_report(report, str(out))
    assert os.path.exists(out)

def test_log_entry_immutability():
    e = LogEntry("2026-08-30 10:00:00", LogLevel.INFO, "Msg")
    with pytest.raises(FrozenInstanceError):
        e.message = "New Msg"  # type: ignore

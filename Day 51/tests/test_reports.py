"""
===============================================================================
DAY 51 — REPORT GENERATOR UNIT TESTS
===============================================================================
This module tests report text formatting, file persistence using pathlib,
and header/summary content validation.
===============================================================================
"""

from pathlib import Path
from typing import List
from app.models import Student
from app.reports import generate_student_report


def test_generate_student_report_creates_file(sample_students: List[Student], tmp_path: Path) -> None:
    """Test report generator creates file at target output path."""
    out_file = tmp_path / "output" / "report.txt"
    report_text = generate_student_report(sample_students, out_file)

    assert out_file.exists()
    assert out_file.is_file()
    assert out_file.read_text(encoding="utf-8") == report_text


def test_generate_student_report_content(sample_students: List[Student], tmp_path: Path) -> None:
    """Test report content contains expected statistical summaries and student details."""
    out_file = tmp_path / "report_test.txt"
    report_text = generate_student_report(sample_students, out_file)

    assert "STUDENT ANALYSIS REPORT" in report_text
    assert "Total Students  : 4" in report_text
    assert "Average Marks   : 80.50" in report_text
    assert "Aisha - 92.00 (Excellent)" in report_text
    assert "Rohan - 67.00 (Average)" in report_text
    assert "Excellent (90+) : 1" in report_text
    assert "Good (75-89)    : 2" in report_text


def test_generate_student_report_empty_students(tmp_path: Path) -> None:
    """Test generating report for empty student list."""
    out_file = tmp_path / "empty_report.txt"
    report_text = generate_student_report([], out_file)

    assert out_file.exists()
    assert "Total Students  : 0" in report_text
    assert "Average Marks   : 0.00" in report_text

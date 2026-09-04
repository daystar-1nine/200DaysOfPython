"""
===============================================================================
DAY 55 — TEST REPORT GENERATOR MODULE
===============================================================================
This test module verifies report formatting and text file creation.
===============================================================================
"""

from pathlib import Path
from app.report import generate_analytics_report
from app.analyzer import analyze_student_performance_v2


def test_generate_analytics_report(tmp_path: Path, sample_small_dataset):
    """Verify generate_analytics_report creates formatted ASCII report file."""
    # What is used: generate_analytics_report with tmp_path fixture.
    # Why it is used: Validates executive ASCII report text output.
    # How it works: Runs analysis on small dataset and asserts text file existence and content strings.
    students, subjects, marks = sample_small_dataset
    analysis = analyze_student_performance_v2(students, subjects, marks)
    report_file = tmp_path / "test_report.txt"

    report_text = generate_analytics_report(analysis, report_file)
    assert report_file.exists()
    assert "STUDENT ANALYTICS REPORT" in report_text
    assert "Class Average:         62.50%" in report_text
    assert "TOP 10 STUDENTS" in report_text

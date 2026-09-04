"""
===============================================================================
DAY 53 — TEST REPORTER MODULE
===============================================================================
This test module verifies report generation and cleaned CSV file exporting.
===============================================================================
"""

from pathlib import Path
from app.reporter import export_cleaned_csv, generate_sales_report


def test_export_cleaned_csv(tmp_path: Path, sample_sales):
    """Verify export_cleaned_csv creates valid CSV output file with header."""
    # What is used: export_cleaned_csv function with sample_sales.
    # Why it is used: Ensures processed clean sales persist to disk in CSV format.
    # How it works: Writes CSV to tmp_path, reads text, and verifies headers and line count.
    out_csv = tmp_path / "cleaned_output.csv"
    export_cleaned_csv(sample_sales, out_csv)
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "order_id,customer,product,category,price,quantity,date,total" in content
    lines = content.strip().splitlines()
    assert len(lines) == 6  # header + 5 rows


def test_generate_sales_report(tmp_path: Path, sample_sales):
    """Verify generate_sales_report creates formatted ASCII report file."""
    # What is used: generate_sales_report function with tmp_path.
    # Why it is used: Validates ASCII summary text report generation.
    # How it works: Checks report file creation and key summary header sections.
    out_report = tmp_path / "sales_report.txt"
    report_text = generate_sales_report(sample_sales, 10, 3, 2, out_report)
    assert out_report.exists()
    assert "SALES ANALYSIS REPORT" in report_text
    assert "Total Orders:         5" in report_text
    assert "Total Revenue:        $88,900.00" in report_text

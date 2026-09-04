"""
Module: report.py
Generates ASCII formatted executive summary reports and exports processed datasets.
"""

# What is used: Import os module and pathlib Path class.
# Why it is used: Manages directory creation and export path operations.
# How it works: Ensures destination directories exist before saving files.
import os
from pathlib import Path

# What is used: Import pandas library.
# Why it is used: Handles CSV export with df.to_csv().
# How it works: Writes DataFrame contents to disk in standard CSV format.
import pandas as pd


def generate_ascii_report(metrics: dict, cleaning_stats: dict) -> str:
    """
    Format analytical metrics into an ASCII executive text report.

    Args:
        metrics: Analytical metrics dictionary from analyzer.
        cleaning_stats: Cleaning audit stats dictionary from cleaner.

    Returns:
        str: Formatted ASCII text report string.
    """
    # What is used: String formatting with f-strings and multiline block construction.
    # Why it is used: Constructs clear, professional ASCII report for terminal output or text export.
    # How it works: Embeds metric variables cleanly inside structured ASCII text borders.
    report_lines = [
        "================================================================================",
        "                  STUDENT PERFORMANCE ANALYZER V2 - EXECUTIVE REPORT            ",
        "================================================================================",
        "",
        "--- DATA CLEANING AUDIT STATS ---",
        f"  * Initial Raw Records   : {cleaning_stats.get('initial_rows', 0)}",
        f"  * Duplicates Removed    : {cleaning_stats.get('duplicates_removed', 0)}",
        f"  * Missing Values Imputed: {cleaning_stats.get('nulls_filled', 0)}",
        f"  * Invalid Scores Dropped: {cleaning_stats.get('invalid_scores_dropped', 0)}",
        f"  * Clean Analytical Rows : {cleaning_stats.get('final_rows', 0)}",
        "",
        "--- CLASS OVERALL METRICS ---",
        f"  * Total Students       : {metrics.get('total_students', 0)}",
        f"  * Passing Students     : {metrics.get('pass_count', 0)} ({metrics.get('pass_rate', 0.0)}%)",
        f"  * Failing Students     : {metrics.get('fail_count', 0)}",
        f"  * Class Mean Score     : {metrics.get('overall_average', 0.0)} / 100",
        ""
    ]

    # Add Top Overall Student
    top_s = metrics.get("top_student", {})
    if top_s:
        report_lines.extend([
            "--- TOP OVERALL PERFORMER ---",
            f"  * Name       : {top_s.get('Name', 'N/A')} ({top_s.get('Student_ID', 'N/A')})",
            f"  * Department : {top_s.get('Department', 'N/A')}",
            f"  * Total Marks: {top_s.get('Total', 0.0)} / 300 (Avg: {top_s.get('Average', 0.0)})",
            ""
        ])

    # Add Subject Toppers
    toppers = metrics.get("subject_toppers", {})
    if toppers:
        report_lines.append("--- SUBJECT TOPPERS ---")
        for subj, info in toppers.items():
            report_lines.append(
                f"  * {subj:<10}: {info.get('Name', 'N/A')} ({info.get('Department', 'N/A')}) - Score: {info.get('Score', 0.0)}"
            )
        report_lines.append("")

    # Add Department Summary
    dept_sum = metrics.get("department_summary", {})
    if dept_sum:
        report_lines.append("--- DEPARTMENT BREAKDOWN ---")
        report_lines.append(f"  {'Dept':<8} {'Count':<8} {'Math':<8} {'Physics':<8} {'Chem':<8} {'Avg':<8} {'Pass Rate':<10}")
        report_lines.append("  " + "-" * 62)
        for dept, d_info in dept_sum.items():
            pass_rate_str = f"{d_info.get('Pass_Rate', 0.0):.2f}%"
            report_lines.append(
                f"  {dept:<8} {d_info.get('Student_Count', 0):<8} "
                f"{d_info.get('Math_Mean', 0.0):<8} {d_info.get('Physics_Mean', 0.0):<8} "
                f"{d_info.get('Chemistry_Mean', 0.0):<8} {d_info.get('Overall_Avg', 0.0):<8} "
                f"{pass_rate_str:<10}"
            )
        report_lines.append("")

    report_lines.append("================================================================================")
    return "\n".join(report_lines)


def export_analysis_results(
    df: pd.DataFrame,
    report_str: str,
    csv_output_path: str | Path,
    report_output_path: str | Path
) -> None:
    """
    Export processed DataFrame to CSV and executive report to text file.

    Args:
        df: Processed student DataFrame.
        report_str: ASCII formatted report string.
        csv_output_path: Destination path for CSV export.
        report_output_path: Destination path for ASCII report text file.
    """
    # What is used: Path.mkdir(parents=True, exist_ok=True).
    # Why it is used: Ensures destination directories exist prior to file creation.
    # How it works: Recursively creates parent directories if missing.
    csv_path = Path(csv_output_path)
    report_path = Path(report_output_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # What is used: df.to_csv(index=False).
    # Why it is used: Writes cleaned and calculated dataset to CSV file without row index column.
    # How it works: Formats DataFrame rows into CSV text and saves to disk.
    df.to_csv(csv_path, index=False)

    # What is used: Standard file open() with encoding='utf-8'.
    # Why it is used: Saves formatted ASCII report to text file safely.
    # How it works: Writes string content to destination file.
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

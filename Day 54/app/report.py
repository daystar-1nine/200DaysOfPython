"""
===============================================================================
DAY 54 — REPORT GENERATOR MODULE
===============================================================================
This module formats student performance analysis results into a structured
ASCII text report saved to the output directory.
===============================================================================
"""

from pathlib import Path
from typing import Dict, Any
from datetime import date


def generate_performance_report(analysis: Dict[str, Any], output_path: Path) -> str:
    """Format analysis dictionary into ASCII text report and save to file."""
    # What is used: Multi-line string formatting and Path file writing.
    # Why it is used: Creates human-readable performance summary report.
    # How it works: Combines student rankings, subject averages, and extrema into ASCII sections.
    today_str = date.today().strftime("%Y-%m-%d")

    lines = [
        "========================================",
        "       STUDENT PERFORMANCE REPORT       ",
        "========================================",
        f"Generated: {today_str}",
        "",
        "CLASS OVERVIEW",
        "----------------------------------------",
        f"Total Students:        {analysis['student_count']}",
        f"Total Subjects:        {analysis['subject_count']}",
        f"Overall Class Average: {analysis['overall_class_average']:.2f}%",
        "",
        "STUDENT RANKINGS",
        "----------------------------------------",
    ]

    for rank, name, avg in analysis["rankings"]:
        lines.append(f"{rank}. {name:<12} -> {avg:.2f}%")

    lines.extend([
        "",
        "SUBJECT PERFORMANCE AVERAGES",
        "----------------------------------------",
    ])

    for subj, avg in analysis["subject_averages"].items():
        lines.append(f"{subj:<15} -> {avg:.2f}%")

    best_st, best_score = analysis["best_student"]
    low_st, low_score = analysis["lowest_student"]
    best_sub, best_sub_score = analysis["best_subject"]

    lines.extend([
        "",
        "EXTREMA & METRICS SUMMARY",
        "----------------------------------------",
        f"Highest Performer:     {best_st} ({best_score:.2f}%)",
        f"Lowest Performer:      {low_st} ({low_score:.2f}%)",
        f"Best Subject:          {best_sub} ({best_sub_score:.2f}%)",
        f"High Performers (>=80): {len(analysis['high_performers'])} students ({', '.join(analysis['high_performers']) if len(analysis['high_performers']) > 0 else 'None'})",
        f"Low Performers (<60):  {len(analysis['low_performers'])} students ({', '.join(analysis['low_performers']) if len(analysis['low_performers']) > 0 else 'None'})",
        "========================================\n",
    ])

    report_content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")

    return report_content

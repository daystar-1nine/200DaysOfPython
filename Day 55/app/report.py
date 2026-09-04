"""
===============================================================================
DAY 55 — REPORT GENERATOR MODULE
===============================================================================
This module formats student analytics results into an executive ASCII text
report saved to the output directory.
===============================================================================
"""

from pathlib import Path
from typing import Dict, Any
from datetime import date


def generate_analytics_report(analysis: Dict[str, Any], output_path: Path) -> str:
    """Format analysis payload into ASCII text report and save to file."""
    # What is used: Multi-line string formatting and Path file writing.
    # Why it is used: Creates human-readable performance summary report.
    # How it works: Combines dataset audit, pass/fail metrics, top 10 rankings, and grade distribution.
    today_str = date.today().strftime("%Y-%m-%d")

    pf = analysis["pass_fail"]
    best_st, best_score = analysis["best_student"]
    low_st, low_score = analysis["lowest_student"]
    best_sub, best_sub_score = analysis["best_subject"]
    low_sub, low_sub_score = analysis["lowest_subject"]

    lines = [
        "============================================",
        "         STUDENT ANALYTICS REPORT           ",
        "============================================",
        f"Generated: {today_str}",
        "",
        "DATASET AUDIT",
        "--------------------------------------------",
        f"Total Students:        {analysis['student_count']}",
        f"Total Subjects:        {analysis['subject_count']}",
        "",
        "OVERALL CLASS STATISTICS",
        "--------------------------------------------",
        f"Class Average:         {analysis['overall_class_average']:.2f}%",
        f"Highest Mark:          {analysis['highest_mark']:.2f}",
        f"Lowest Mark:           {analysis['lowest_mark']:.2f}",
        "",
        "PERFORMANCE METRICS",
        "--------------------------------------------",
        f"Passed Students:       {pf['pass_count']} ({pf['pass_percentage']:.1f}%)",
        f"Failed Students:       {pf['fail_count']} ({pf['fail_percentage']:.1f}%)",
        f"High Performers (>=80):{analysis['high_performers_count']} students",
        f"Low Performers (<40):  {analysis['low_performers_count']} students",
        "",
        "SUBJECT PERFORMANCE AVERAGES",
        "--------------------------------------------",
    ]

    for subj, avg in analysis["subject_averages"].items():
        lines.append(f"{subj:<15} -> {avg:.2f}%")

    lines.extend([
        "",
        "SUBJECT EXTREMA",
        "--------------------------------------------",
        f"Best Subject:          {best_sub} ({best_sub_score:.2f}%)",
        f"Lowest Subject:        {low_sub} ({low_sub_score:.2f}%)",
        "",
        "TOP 10 STUDENTS",
        "--------------------------------------------",
    ])

    for rank, name, avg in analysis["top_10_students"]:
        lines.append(f"{rank:2d}. {name:<12} -> {avg:.2f}%")

    lines.extend([
        "",
        "GRADE DISTRIBUTION",
        "--------------------------------------------",
    ])

    for grade, count in analysis["grade_distribution"].items():
        lines.append(f"{grade:<3} -> {count} students")

    lines.extend([
        "",
        "STUDENT EXTREMA",
        "--------------------------------------------",
        f"Highest Performer:     {best_st} ({best_score:.2f}%)",
        f"Lowest Performer:      {low_st} ({low_score:.2f}%)",
        "============================================\n",
    ])

    report_content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")

    return report_content

"""
===============================================================================
DAY 51 — REPORT GENERATOR MODULE (PATHLIB IO)
===============================================================================
This module formats student analysis reports and persists output files using pathlib.
===============================================================================
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from app.models import Student
from app.enums import PerformanceLevel
from app.services import (
    calculate_average,
    get_highest_scorer,
    get_lowest_scorer,
    count_students,
    get_performance_distribution,
)


def generate_student_report(students: List[Student], output_path: Path) -> str:
    """Generate formatted ASCII report string and save to output_path using pathlib."""
    # What is used: Path.parent.mkdir() directory creation.
    # Why it is used: Ensures destination directory exists before writing file.
    # How it works: Recursively creates output parent directories if needed.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total = count_students(students)
    avg = calculate_average(students)
    highest = get_highest_scorer(students) if students else None
    lowest = get_lowest_scorer(students) if students else None
    dist = get_performance_distribution(students)

    lines = [
        "========================================",
        "          STUDENT ANALYSIS REPORT       ",
        "========================================",
        f"Generated: {timestamp}",
        "",
        f"Total Students  : {total}",
        f"Average Marks   : {avg:.2f}",
        "",
    ]

    if highest and lowest:
        lines.extend([
            "Highest Scorer:",
            f"  {highest.name} - {highest.marks:.2f} ({highest.performance_level.value})",
            "",
            "Lowest Scorer:",
            f"  {lowest.name} - {lowest.marks:.2f} ({lowest.performance_level.value})",
            "",
        ])

    lines.extend([
        "Performance Distribution:",
        f"  Excellent (90+) : {dist.get(PerformanceLevel.EXCELLENT, 0)}",
        f"  Good (75-89)    : {dist.get(PerformanceLevel.GOOD, 0)}",
        f"  Average (50-74) : {dist.get(PerformanceLevel.AVERAGE, 0)}",
        f"  Poor (<50)      : {dist.get(PerformanceLevel.POOR, 0)}",
        "========================================",
    ])

    report_content = "\n".join(lines)

    # What is used: Path.write_text() string persistence.
    # Why it is used: Writes complete report string to disk cleanly.
    # How it works: Encodes string to UTF-8 and writes to output_path.
    output_path.write_text(report_content, encoding="utf-8")

    return report_content

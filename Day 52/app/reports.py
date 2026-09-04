"""
===============================================================================
DAY 52 — REPORT GENERATOR MODULE
===============================================================================
This module formats student summary reports with course distribution counts using
collections.Counter and persists output to disk via pathlib.Path.
===============================================================================
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from app.models import Student
from app.services import (
    calculate_average,
    get_highest_scorer,
    get_lowest_scorer,
)


def generate_student_report(students: List[Student], output_path: Path) -> str:
    """Generate ASCII summary report and save file via pathlib.Path."""
    # What is used: Path.parent.mkdir and collections.Counter.
    # Why it is used: Creates directory, tallies course counts, and writes report text.
    # How it works: Tallies course distribution using Counter(s.course for s in students).
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total = len(students)
    avg = calculate_average(students)
    highest = get_highest_scorer(students) if students else None
    lowest = get_lowest_scorer(students) if students else None

    # Tally course distribution using Counter
    course_counts = Counter(s.course for s in students)

    lines = [
        "===================================",
        "       STUDENT DATA REPORT         ",
        "===================================",
        f"Generated: {timestamp}",
        "",
        f"Total Students: {total}",
        f"Average Marks : {avg:.2f}",
        "",
    ]

    if highest and lowest:
        lines.extend([
            "Highest Scorer:",
            f"  {highest.name} - {highest.marks:.2f} ({highest.course})",
            "",
            "Lowest Scorer:",
            f"  {lowest.name} - {lowest.marks:.2f} ({lowest.course})",
            "",
        ])

    lines.append("Courses Breakdown:")
    if course_counts:
        for course, count in sorted(course_counts.items()):
            lines.append(f"  {course}: {count}")
    else:
        lines.append("  No courses recorded.")

    lines.append("===================================")

    report_content = "\n".join(lines)
    output_path.write_text(report_content, encoding="utf-8")
    return report_content

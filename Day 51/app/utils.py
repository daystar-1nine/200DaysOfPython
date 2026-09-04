"""
===============================================================================
DAY 51 — UTILITY HELPER MODULE
===============================================================================
This module provides display formatting and table utilities for the CLI.
===============================================================================
"""

from typing import List
from app.models import Student


def format_student_table(students: List[Student]) -> str:
    """Format student list as tabular ASCII string."""
    # What is used: String formatting with fixed width padding.
    # Why it is used: Displays student records neatly in terminal console.
    # How it works: Formats header row, separator line, and padded student rows.
    if not students:
        return "No student records available."

    lines = [
        f"{'Name':<15} | {'Age':<5} | {'Marks':<7} | {'Level':<12}",
        "-" * 48,
    ]
    for s in students:
        lines.append(f"{s.name:<15} | {s.age:<5} | {s.marks:<7.2f} | {s.performance_level.value:<12}")
    return "\n".join(lines)

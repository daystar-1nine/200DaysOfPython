"""
===============================================================================
DAY 52 — UTILITY DISPLAY HELPER MODULE
===============================================================================
This module provides tabular ASCII formatting utilities for displaying student lists.
===============================================================================
"""

from typing import List
from app.models import Student


def format_student_table(students: List[Student]) -> str:
    """Format student records into tabular ASCII string for CLI output."""
    # What is used: Fixed-width string formatting.
    # Why it is used: Displays student fields neatly in terminal.
    # How it works: Formats header row, separator line, and formatted student records.
    if not students:
        return "No student records available."

    lines = [
        f"{'ID':<5} | {'Name':<15} | {'Age':<5} | {'Course':<20} | {'Marks':<7}",
        "-" * 62,
    ]
    for s in students:
        lines.append(f"{s.id:<5} | {s.name:<15} | {s.age:<5} | {s.course:<20} | {s.marks:<7.2f}")
    return "\n".join(lines)

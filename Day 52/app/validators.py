"""
===============================================================================
DAY 52 — DATA VALIDATION MODULE
===============================================================================
This module provides validation logic for Student dataclass instances before
persistence or processing.
===============================================================================
"""

from app.models import Student


def validate_student(student: Student) -> None:
    """Validate student fields enforcing domain integrity rules."""
    # What is used: Explicit attribute validation checks.
    # Why it is used: Ensures untrusted external inputs adhere to constraints.
    # How it works: Raises ValueError if name is empty or age/marks fall outside valid bounds.
    if not isinstance(student.id, int) or student.id <= 0:
        raise ValueError(f"Student ID must be a positive integer. Got: {student.id}")

    if not isinstance(student.name, str) or not student.name.strip():
        raise ValueError("Student name cannot be empty or whitespace only.")

    if not isinstance(student.age, int) or not (1 <= student.age <= 100):
        raise ValueError(f"Invalid age '{student.age}'. Age must be an integer between 1 and 100.")

    if not isinstance(student.course, str) or not student.course.strip():
        raise ValueError("Course name cannot be empty.")

    if not isinstance(student.marks, (int, float)) or not (0.0 <= float(student.marks) <= 100.0):
        raise ValueError(f"Invalid marks '{student.marks}'. Marks must be a number between 0 and 100.")

"""
===============================================================================
DAY 52 — TEST VALIDATORS MODULE
===============================================================================
This test module verifies student model field validation, ensuring domain rules
are enforced for ID, name, age, course, and marks fields.
===============================================================================
"""

import pytest
from app.models import Student
from app.validators import validate_student


def test_validate_student_valid(single_student):
    """Verify that a valid Student object passes validation without error."""
    # What is used: pytest assertion with validate_student function.
    # Why it is used: Ensures valid model inputs pass domain validation rules cleanly.
    # How it works: Passes single_student fixture instance; expects no exception.
    validate_student(single_student[0])


def test_validate_student_invalid_id():
    """Verify that non-positive or non-integer student ID raises ValueError."""
    # What is used: pytest.raises context manager.
    # Why it is used: Enforces positive integer constraint on Student ID field.
    # How it works: Checks negative ID (-1) and zero ID (0) raise ValueError.
    with pytest.raises(ValueError, match="Student ID must be a positive integer"):
        validate_student(Student(-1, "Rahul", 20, "CS", 80.0))

    with pytest.raises(ValueError, match="Student ID must be a positive integer"):
        validate_student(Student(0, "Rahul", 20, "CS", 80.0))


def test_validate_student_invalid_name():
    """Verify that empty or whitespace-only student names raise ValueError."""
    # What is used: pytest.raises with empty and whitespace strings.
    # Why it is used: Prevents creation of anonymous or corrupt student names.
    # How it works: Instantiates Student with "" or "   " and verifies ValueError.
    with pytest.raises(ValueError, match="Student name cannot be empty"):
        validate_student(Student(1, "", 20, "CS", 80.0))

    with pytest.raises(ValueError, match="Student name cannot be empty"):
        validate_student(Student(1, "   ", 20, "CS", 80.0))


def test_validate_student_invalid_age():
    """Verify that out-of-range student ages raise ValueError."""
    # What is used: Boundary testing for age attribute (0 and 101).
    # Why it is used: Validates domain constraint requiring age between 1 and 100.
    # How it works: Expects ValueError when age < 1 or age > 100.
    with pytest.raises(ValueError, match="Invalid age"):
        validate_student(Student(1, "Rahul", 0, "CS", 80.0))

    with pytest.raises(ValueError, match="Invalid age"):
        validate_student(Student(1, "Rahul", 101, "CS", 80.0))


def test_validate_student_invalid_course():
    """Verify that empty course strings raise ValueError."""
    # What is used: pytest.raises for course name validation.
    # Why it is used: Ensures every student belongs to a valid course department.
    # How it works: Validates whitespace and empty string raise ValueError.
    with pytest.raises(ValueError, match="Course name cannot be empty"):
        validate_student(Student(1, "Rahul", 20, "", 80.0))


def test_validate_student_invalid_marks():
    """Verify that out-of-range marks raise ValueError."""
    # What is used: Boundary testing for marks attribute (-5.0 and 105.0).
    # Why it is used: Ensures student marks fall within valid academic percentage range (0-100).
    # How it works: Expects ValueError when marks < 0.0 or marks > 100.0.
    with pytest.raises(ValueError, match="Invalid marks"):
        validate_student(Student(1, "Rahul", 20, "CS", -5.0))

    with pytest.raises(ValueError, match="Invalid marks"):
        validate_student(Student(1, "Rahul", 20, "CS", 105.0))

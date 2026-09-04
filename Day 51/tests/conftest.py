"""
===============================================================================
DAY 51 — PYTEST FIXTURES (TEST DATA & TEMPORARY FILES)
===============================================================================
This module provides Pytest fixtures for student dataclass instances, temporary
CSV files, and output report paths using tmp_path.
===============================================================================
"""

import pytest
from pathlib import Path
from typing import List
from app.models import Student


@pytest.fixture
def sample_students() -> List[Student]:
    """Fixture providing a list of test Student dataclass instances."""
    return [
        Student("Rahul", 21, 85.0),
        Student("Aisha", 20, 92.0),
        Student("Rohan", 22, 67.0),
        Student("Sneha", 21, 78.0),
    ]


@pytest.fixture
def single_student() -> List[Student]:
    """Fixture providing a list with a single student."""
    return [Student("Alone", 20, 95.0)]


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a valid temporary CSV file."""
    csv_file = tmp_path / "students_valid.csv"
    csv_file.write_text("name,age,marks\nRahul,21,85\nAisha,20,92\nRohan,22,67\n", encoding="utf-8")
    return csv_file


@pytest.fixture
def empty_csv_file(tmp_path: Path) -> Path:
    """Fixture creating an empty temporary CSV file."""
    csv_file = tmp_path / "students_empty.csv"
    csv_file.write_text("", encoding="utf-8")
    return csv_file


@pytest.fixture
def malformed_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a CSV file with invalid numeric marks."""
    csv_file = tmp_path / "students_malformed.csv"
    csv_file.write_text("name,age,marks\nRahul,21,invalid_marks\n", encoding="utf-8")
    return csv_file


@pytest.fixture
def missing_header_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a CSV missing required headers."""
    csv_file = tmp_path / "students_no_header.csv"
    csv_file.write_text("wrong1,wrong2\nval1,val2\n", encoding="utf-8")
    return csv_file

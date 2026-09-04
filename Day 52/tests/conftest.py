"""
===============================================================================
DAY 52 — PYTEST FIXTURES (TEST DATA & TEMPORARY FILES)
===============================================================================
This module provides Pytest fixtures for Student dataclass records, temporary
JSON files, temporary CSV files, and report output paths using tmp_path.
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
        Student(1, "Rahul", 21, "Data Science", 85.0),
        Student(2, "Aisha", 20, "Computer Science", 95.0),
        Student(3, "Rohan", 22, "Data Science", 67.0),
        Student(4, "Sneha", 21, "Computer Science", 78.0),
        Student(5, "Karan", 23, "AI & Robotics", 82.0),
    ]


@pytest.fixture
def single_student() -> List[Student]:
    """Fixture providing a list with a single student."""
    return [Student(1, "Sole", 20, "Math", 90.0)]


@pytest.fixture
def valid_json_file(tmp_path: Path) -> Path:
    """Fixture creating a valid temporary JSON file."""
    json_file = tmp_path / "test_students.json"
    content = """[
        {"id": 1, "name": "Rahul", "age": 21, "course": "Data Science", "marks": 85.0},
        {"id": 2, "name": "Aisha", "age": 20, "course": "Computer Science", "marks": 95.0}
    ]"""
    json_file.write_text(content, encoding="utf-8")
    return json_file


@pytest.fixture
def invalid_json_file(tmp_path: Path) -> Path:
    """Fixture creating a malformed temporary JSON file."""
    json_file = tmp_path / "test_invalid.json"
    json_file.write_text("{malformed_json: true,", encoding="utf-8")
    return json_file


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a valid temporary CSV file."""
    csv_file = tmp_path / "test_students.csv"
    content = "id,name,age,course,marks\n1,Rahul,21,Data Science,85.0\n2,Aisha,20,Computer Science,95.0\n"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture
def invalid_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a CSV file with malformed numeric row data."""
    csv_file = tmp_path / "test_invalid.csv"
    content = "id,name,age,course,marks\n1,Rahul,not_an_age,Data Science,85.0\n"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file

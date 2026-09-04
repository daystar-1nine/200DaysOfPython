"""
===============================================================================
DAY 51 — FILE HANDLER UNIT TESTS
===============================================================================
This module tests CSV reading, parsing, missing file handling, and malformed CSV validation.
===============================================================================
"""

import pytest
from pathlib import Path
from app.file_handler import read_students_csv


def test_read_valid_csv_file(valid_csv_file: Path) -> None:
    """Test reading and parsing a valid CSV file."""
    students = read_students_csv(valid_csv_file)
    assert len(students) == 3
    assert students[0].name == "Rahul"
    assert students[0].age == 21
    assert students[0].marks == 85.0


def test_read_missing_csv_file_raises(tmp_path: Path) -> None:
    """Test reading a non-existent CSV file raises FileNotFoundError."""
    missing = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        read_students_csv(missing)


def test_read_empty_csv_file_raises(empty_csv_file: Path) -> None:
    """Test reading an empty CSV file raises ValueError."""
    with pytest.raises(ValueError, match="is empty"):
        read_students_csv(empty_csv_file)


def test_read_malformed_csv_file_raises(malformed_csv_file: Path) -> None:
    """Test reading a CSV file with invalid numeric marks raises ValueError."""
    with pytest.raises(ValueError, match="Invalid marks value"):
        read_students_csv(malformed_csv_file)


def test_read_missing_header_csv_file_raises(missing_header_csv_file: Path) -> None:
    """Test reading a CSV file with missing headers raises ValueError."""
    with pytest.raises(ValueError, match="missing required headers"):
        read_students_csv(missing_header_csv_file)

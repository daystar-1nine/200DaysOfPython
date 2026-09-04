"""
===============================================================================
DAY 52 — TEST CSV HANDLER MODULE
===============================================================================
This test module verifies save_students_csv and load_students_csv functions,
testing roundtrip CSV export/import, header validation, and invalid row handling.
===============================================================================
"""

import pytest
from pathlib import Path
from app.models import Student
from app.csv_handler import save_students_csv, load_students_csv


def test_save_and_load_students_csv(tmp_path: Path, sample_students):
    """Verify roundtrip CSV serialization and deserialization."""
    # What is used: save_students_csv and load_students_csv with tmp_path.
    # Why it is used: Ensures dataclass objects serialize to CSV format and reload accurately.
    # How it works: Writes CSV file, reads CSV back, and checks length and attribute values.
    csv_path = tmp_path / "students.csv"
    save_students_csv(sample_students, csv_path)
    assert csv_path.exists()

    loaded = load_students_csv(csv_path)
    assert len(loaded) == len(sample_students)
    assert loaded[0].id == sample_students[0].id
    assert loaded[0].name == sample_students[0].name
    assert loaded[1].course == sample_students[1].course


def test_load_students_valid_csv_fixture(valid_csv_file):
    """Verify loading from valid_csv_file fixture."""
    # What is used: valid_csv_file fixture.
    # Why it is used: Tests parsing of standard CSV rows with headers.
    # How it works: Deserializes valid CSV fixture file and checks expected attributes.
    loaded = load_students_csv(valid_csv_file)
    assert len(loaded) == 2
    assert loaded[0].name == "Rahul"
    assert loaded[1].marks == 95.0


def test_load_students_csv_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError raised when CSV file path does not exist."""
    # What is used: pytest.raises with FileNotFoundError.
    # Why it is used: Enforces path existence checks for CSV files.
    # How it works: Calls load_students_csv with non-existent path.
    non_existent = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        load_students_csv(non_existent)


def test_load_students_csv_invalid_header(tmp_path: Path):
    """Verify ValueError raised when CSV headers are missing required columns."""
    # What is used: tmp_path to create CSV with incomplete column header row.
    # Why it is used: Ensures CSV column schemas match required model fields.
    # How it works: Creates CSV with missing 'marks' header; asserts ValueError.
    bad_header_file = tmp_path / "bad_headers.csv"
    bad_header_file.write_text("id,name,age\n1,Rahul,21\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV missing required column headers"):
        load_students_csv(bad_header_file)


def test_load_students_csv_invalid_row(invalid_csv_file):
    """Verify ValueError raised when CSV row contains malformed data types."""
    # What is used: invalid_csv_file fixture with non-numeric age value.
    # Why it is used: Validates type conversion and validation handling per row.
    # How it works: Expects ValueError when int("not_an_age") conversion fails.
    with pytest.raises(ValueError, match="Invalid student record in CSV"):
        load_students_csv(invalid_csv_file)

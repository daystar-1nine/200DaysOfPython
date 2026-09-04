"""
===============================================================================
DAY 52 — TEST JSON HANDLER MODULE
===============================================================================
This test module verifies save_students and load_students JSON IO functions,
testing serialization, deserialization, missing files, and malformed JSON format.
===============================================================================
"""

import pytest
from pathlib import Path
from app.models import Student
from app.json_handler import save_students, load_students


def test_save_and_load_students_json(tmp_path: Path, sample_students):
    """Verify roundtrip JSON serialization and deserialization."""
    # What is used: save_students and load_students with tmp_path fixture.
    # Why it is used: Ensures dataclass instances serialize to JSON file and reload identically.
    # How it works: Saves sample_students to temp file, reloads, and compares list equality.
    json_path = tmp_path / "students.json"
    save_students(sample_students, json_path)
    assert json_path.exists()

    loaded = load_students(json_path)
    assert len(loaded) == len(sample_students)
    assert loaded[0].id == sample_students[0].id
    assert loaded[0].name == sample_students[0].name
    assert loaded[0].course == sample_students[0].course


def test_load_students_valid_fixture(valid_json_file):
    """Verify loading from valid_json_file Pytest fixture."""
    # What is used: valid_json_file fixture.
    # Why it is used: Tests loading pre-written valid JSON file content.
    # How it works: Deserializes valid fixture JSON and verifies model attribute fields.
    loaded = load_students(valid_json_file)
    assert len(loaded) == 2
    assert loaded[0].name == "Rahul"
    assert loaded[1].name == "Aisha"


def test_load_students_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError raised when loading non-existent JSON file path."""
    # What is used: pytest.raises with FileNotFoundError.
    # Why it is used: Enforces defensive exception handling for missing files.
    # How it works: Passes non-existent Path to load_students; expects FileNotFoundError.
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="JSON file not found"):
        load_students(non_existent)


def test_load_students_malformed_json(invalid_json_file):
    """Verify ValueError raised when parsing malformed JSON text content."""
    # What is used: pytest.raises with ValueError and JSON error matching.
    # Why it is used: Validates defensive exception catching for json.JSONDecodeError.
    # How it works: Attempts loading invalid_json_file fixture; asserts ValueError is raised.
    with pytest.raises(ValueError, match="Failed to parse JSON file"):
        load_students(invalid_json_file)


def test_load_students_empty_file(tmp_path: Path):
    """Verify loading empty JSON file returns empty list."""
    # What is used: Empty file creation via write_text("").
    # Why it is used: Handles edge case where JSON file exists but is zero-byte empty.
    # How it works: Returns [] without raising JSON decode exception.
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    assert load_students(empty_file) == []

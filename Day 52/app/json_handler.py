"""
===============================================================================
DAY 52 — JSON SERIALIZATION HANDLER MODULE
===============================================================================
This module provides save_students and load_students for JSON file IO with
defensive exception handling for missing or malformed JSON data.
===============================================================================
"""

import json
from pathlib import Path
from typing import List
from app.models import Student
from app.validators import validate_student


def save_students(students: List[Student], path: Path) -> None:
    """Serialize a list of Student dataclass instances to a JSON file."""
    # What is used: Path.parent.mkdir and json.dump with indent=4.
    # Why it is used: Writes formatted JSON array file to disk.
    # How it works: Converts Student objects to dicts via to_dict() and writes JSON stream.
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_list = [s.to_dict() for s in students]
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict_list, f, indent=4)


def load_students(path: Path) -> List[Student]:
    """Read a JSON file and deserialize back into Student dataclass objects."""
    # What is used: Path.exists check and json.load with JSONDecodeError handling.
    # Why it is used: Safely loads JSON file contents and converts entries to Student models.
    # How it works: Reads JSON file; raises FileNotFoundError or ValueError on malformed JSON.
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"JSON file not found at path: '{path}'")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file at '{path}': Malformed format. {e}")

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}', got {type(raw_data).__name__}.")

    students: List[Student] = []
    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(f"JSON array element at index {idx} is not an object.")
        try:
            student = Student(
                id=int(item["id"]),
                name=str(item["name"]),
                age=int(item["age"]),
                course=str(item["course"]),
                marks=float(item["marks"]),
            )
            validate_student(student)
            students.append(student)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid student record at index {idx}: {e}")

    return students

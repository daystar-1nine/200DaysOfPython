"""
===============================================================================
DAY 52 — CSV SERIALIZATION HANDLER MODULE
===============================================================================
This module provides save_students_csv and load_students_csv for CSV file IO
using csv.DictWriter and csv.DictReader with defensive validation.
===============================================================================
"""

import csv
from pathlib import Path
from typing import List
from app.models import Student
from app.validators import validate_student


def save_students_csv(students: List[Student], path: Path) -> None:
    """Serialize a list of Student dataclass instances to a CSV file."""
    # What is used: Path.parent.mkdir and csv.DictWriter.
    # Why it is used: Writes headers and student dictionary rows to CSV file stream.
    # How it works: Writes header row via writeheader() and iterates student dict payloads.
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "name", "age", "course", "marks"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for student in students:
            writer.writerow(student.to_dict())


def load_students_csv(path: Path) -> List[Student]:
    """Read a CSV file and parse rows into Student dataclass objects."""
    # What is used: Path.exists check and csv.DictReader context manager.
    # Why it is used: Deserializes CSV text rows into typed Student model instances.
    # How it works: Validates headers, converts cell strings to int/float, validates Student.
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"CSV file not found at path: '{path}'")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    students: List[Student] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"id", "name", "age", "course", "marks"}
        if not reader.fieldnames or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required column headers: {required_fields}")

        for line_num, row in enumerate(reader, start=2):
            try:
                student = Student(
                    id=int(row["id"].strip()),
                    name=row["name"].strip(),
                    age=int(row["age"].strip()),
                    course=row["course"].strip(),
                    marks=float(row["marks"].strip()),
                )
                validate_student(student)
                students.append(student)
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"Row {line_num}: Invalid student record in CSV: {e}")

    return students

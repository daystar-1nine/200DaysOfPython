"""
===============================================================================
DAY 51 — FILE HANDLER MODULE (PATHLIB & CSV PARSER)
===============================================================================
This module handles reading and parsing CSV files using pathlib.Path and csv.DictReader
with validation for missing files, empty rows, and malformed data types.
===============================================================================
"""

import csv
from pathlib import Path
from typing import List
from app.models import Student
from app.enums import get_performance_level


def read_students_csv(file_path: Path) -> List[Student]:
    """Read a CSV file containing student records and parse into Student objects."""
    # What is used: Path.exists() existence check.
    # Why it is used: Validates target file presence before attempting file read.
    # How it works: Raises FileNotFoundError if file_path is missing or not a file.
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found at path: '{file_path}'")

    # What is used: Path.read_text() content check.
    # Why it is used: Detects empty CSV files immediately.
    # How it works: Reads raw file text and checks if stripped content is empty.
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"CSV file at '{file_path}' is empty.")

    students: List[Student] = []

    # What is used: Path.open() with csv.DictReader context manager.
    # Why it is used: Safely parses CSV rows into dictionaries using header keys.
    # How it works: Opens file stream, validates headers (name, age, marks), converts types.
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_headers = {"name", "age", "marks"}
        if not reader.fieldnames or not required_headers.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required headers: {required_headers}")

        for line_num, row in enumerate(reader, start=2):
            name = row.get("name", "").strip()
            age_raw = row.get("age", "").strip()
            marks_raw = row.get("marks", "").strip()

            if not name:
                raise ValueError(f"Row {line_num}: Missing student name.")

            try:
                age = int(age_raw)
                if age <= 0:
                    raise ValueError(f"Row {line_num}: Age must be positive integer.")
            except ValueError as e:
                raise ValueError(f"Row {line_num}: Invalid age value '{age_raw}'. {e}")

            try:
                marks = float(marks_raw)
                # Triggers bounds check
                get_performance_level(marks)
            except ValueError as e:
                raise ValueError(f"Row {line_num}: Invalid marks value '{marks_raw}'. {e}")

            students.append(Student(name=name, age=age, marks=marks))

    if not students:
        raise ValueError(f"No valid student data rows found in '{file_path}'.")

    return students

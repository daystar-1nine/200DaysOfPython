"""
===============================================================================
DAY 51 — CODING CHALLENGE 2: HIGHEST SCORER FROM DATACLASS LIST
===============================================================================
This module defines a Student dataclass and extracts the student with the highest
marks from a list using max(key=...).
===============================================================================
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Student:
    """Dataclass representing student academic details."""
    name: str
    age: int
    marks: float


def get_highest_student(students: List[Student]) -> Student:
    """Find and return the student object with the highest marks."""
    # What is used: max() function with key parameter lambda extractor.
    # Why it is used: Returns the object instance maximizing student.marks.
    # How it works: Iterates students list and compares marks attribute.
    if not students:
        raise ValueError("Student list cannot be empty.")
    return max(students, key=lambda s: s.marks)


if __name__ == "__main__":
    sample_students = [
        Student("A", 20, 80.0),
        Student("B", 21, 95.0),
        Student("C", 20, 88.0),
    ]
    top_student = get_highest_student(sample_students)
    result_str = f"{top_student.name} - {int(top_student.marks)}"
    print("Highest Scorer:", result_str)
    assert result_str == "B - 95"
    print("[OK] Challenge 2 Passed!")

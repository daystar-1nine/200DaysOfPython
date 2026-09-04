"""
===============================================================================
DAY 52 — STUDENT BUSINESS LOGIC SERVICES MODULE
===============================================================================
This module executes student CRUD operations, unique ID generation, multi-field
search filtering, and statistical analysis computations.
===============================================================================
"""

from typing import List
from app.models import Student
from app.validators import validate_student


def generate_next_id(students: List[Student]) -> int:
    """Generate a unique incrementing student ID."""
    # What is used: Generator expression inside max() function with default=0 fallback.
    # Why it is used: Ensures newly added student receives a unique incrementing ID integer.
    # How it works: Evaluates max(student.id) across list and adds 1.
    return max((s.id for s in students), default=0) + 1


def add_student(students: List[Student], new_student: Student) -> Student:
    """Add a new student enforcing unique ID and data validation."""
    # What is used: Validation check and unique ID collision check.
    # Why it is used: Prevents adding corrupt or duplicate ID student records.
    # How it works: Validates fields via validate_student and checks ID uniqueness.
    validate_student(new_student)
    if any(s.id == new_student.id for s in students):
        raise ValueError(f"Student with ID {new_student.id} already exists.")

    students.append(new_student)
    return new_student


def update_student(
    students: List[Student],
    student_id: int,
    name: str | None = None,
    age: int | None = None,
    course: str | None = None,
    marks: float | None = None,
) -> Student:
    """Update fields of an existing student by ID."""
    # What is used: Student lookup by ID and field attribute assignment.
    # Why it is used: Modifies student attributes and runs validate_student checks.
    # How it works: Finds matching Student instance or raises KeyError if not found.
    student = next((s for s in students if s.id == student_id), None)
    if not student:
        raise KeyError(f"Student with ID {student_id} not found.")

    if name is not None:
        student.name = name
    if age is not None:
        student.age = age
    if course is not None:
        student.course = course
    if marks is not None:
        student.marks = marks

    validate_student(student)
    return student


def delete_student(students: List[Student], student_id: int) -> Student:
    """Delete student record by ID."""
    # What is used: List index lookup and pop.
    # Why it is used: Removes matching student from list container.
    # How it works: Removes matching Student instance or raises KeyError.
    idx = next((i for i, s in enumerate(students) if s.id == student_id), None)
    if idx is None:
        raise KeyError(f"Student with ID {student_id} not found.")

    return students.pop(idx)


def search_students(students: List[Student], query: str) -> List[Student]:
    """Search students by ID integer string, name substring, or course substring."""
    # What is used: Filter list comprehension with lower-case substring matching.
    # Why it is used: Performs multi-field search against ID, name, or course.
    # How it works: Returns list of Student objects matching query string.
    q = query.strip().lower()
    if not q:
        return students

    results = []
    for s in students:
        if str(s.id) == q or q in s.name.lower() or q in s.course.lower():
            results.append(s)
    return results


def calculate_average(students: List[Student]) -> float:
    """Calculate mean average marks across all students."""
    if not students:
        return 0.0
    return sum(s.marks for s in students) / len(students)


def get_highest_scorer(students: List[Student]) -> Student:
    """Find student with highest marks."""
    if not students:
        raise ValueError("Cannot find highest scorer in empty student list.")
    return max(students, key=lambda s: s.marks)


def get_lowest_scorer(students: List[Student]) -> Student:
    """Find student with lowest marks."""
    if not students:
        raise ValueError("Cannot find lowest scorer in empty student list.")
    return min(students, key=lambda s: s.marks)

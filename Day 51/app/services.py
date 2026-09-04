"""
===============================================================================
DAY 51 — STUDENT ANALYSIS SERVICE LAYER
===============================================================================
This module provides analytics utilities for computing average marks, identifying
highest/lowest scorers, counting students, filtering by threshold, and performance tallying.
===============================================================================
"""

from collections import Counter
from typing import List, Dict
from app.models import Student
from app.enums import PerformanceLevel


def calculate_average(students: List[Student]) -> float:
    """Calculate mean average marks across all students."""
    # What is used: Generator sum and len list operations.
    # Why it is used: Computes arithmetic mean marks score safely.
    # How it works: Returns sum(s.marks for s in students) / len(students) or 0.0 if empty.
    if not students:
        return 0.0
    total = sum(s.marks for s in students)
    return total / len(students)


def get_highest_scorer(students: List[Student]) -> Student:
    """Find student with highest marks."""
    # What is used: max() function with key=lambda s: s.marks.
    # Why it is used: Extracts Student dataclass instance with highest marks score.
    # How it works: Raises ValueError if student list is empty.
    if not students:
        raise ValueError("Cannot find highest scorer in empty student list.")
    return max(students, key=lambda s: s.marks)


def get_lowest_scorer(students: List[Student]) -> Student:
    """Find student with lowest marks."""
    # What is used: min() function with key=lambda s: s.marks.
    # Why it is used: Extracts Student dataclass instance with lowest marks score.
    # How it works: Raises ValueError if student list is empty.
    if not students:
        raise ValueError("Cannot find lowest scorer in empty student list.")
    return min(students, key=lambda s: s.marks)


def count_students(students: List[Student]) -> int:
    """Return total count of students."""
    return len(students)


def filter_students_by_marks(students: List[Student], min_marks: float) -> List[Student]:
    """Filter students having marks >= min_marks threshold."""
    # What is used: List comprehension with score threshold filtering predicate.
    # Why it is used: Filters dataset matching minimum score requirement.
    # How it works: Returns list of Student instances with marks >= min_marks.
    return [s for s in students if s.marks >= min_marks]


def get_performance_distribution(students: List[Student]) -> Dict[PerformanceLevel, int]:
    """Tally student performance level distribution using collections.Counter."""
    # What is used: collections.Counter over performance_level properties.
    # Why it is used: Tallies occurrences for each PerformanceLevel enum value.
    # How it works: Iterates performance_level property of each student and builds dict.
    distribution = Counter(s.performance_level for s in students)
    # Ensure all enum keys exist in return dictionary
    result: Dict[PerformanceLevel, int] = {}
    for level in PerformanceLevel:
        result[level] = distribution.get(level, 0)
    return result

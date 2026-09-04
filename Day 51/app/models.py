"""
===============================================================================
DAY 51 — STUDENT DATACLASS MODEL
===============================================================================
This module defines the Student dataclass representing student data records.
===============================================================================
"""

from dataclasses import dataclass
from app.enums import PerformanceLevel, get_performance_level


@dataclass
class Student:
    """Dataclass representing student data attributes and performance computation."""

    # What is used: Dataclass field type annotations.
    # Why it is used: Defines student record schema cleanly without boilerplate __init__.
    # How it works: Automatically generates __init__, __repr__, and __eq__ methods.
    name: str
    age: int
    marks: float

    @property
    def performance_level(self) -> PerformanceLevel:
        """Computed property returning performance level classification."""
        # What is used: Dynamic property delegate method.
        # Why it is used: Computes performance enum on-the-fly based on current marks.
        # How it works: Calls get_performance_level(self.marks).
        return get_performance_level(self.marks)

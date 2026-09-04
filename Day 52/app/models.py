"""
===============================================================================
DAY 52 — STUDENT DATACLASS MODEL DEFINITION
===============================================================================
This module defines the Student dataclass representing student entity records
with id, name, age, course, and marks fields.
===============================================================================
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class Student:
    """Dataclass representing a student entity."""

    # What is used: Dataclass field type annotations.
    # Why it is used: Encapsulates student data schema without boilerplate methods.
    # How it works: Generates __init__, __repr__, and __eq__ methods automatically.
    id: int
    name: str
    age: int
    course: str
    marks: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert Student instance to dictionary payload."""
        # What is used: dataclasses.asdict utility.
        # Why it is used: Serializes Dataclass instance to standard Python dict.
        # How it works: Recursively converts dataclass fields to dict key-values.
        return asdict(self)

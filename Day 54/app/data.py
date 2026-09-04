"""
===============================================================================
DAY 54 — DATASET & VALIDATION MODULE
===============================================================================
This module provides sample student marks matrix datasets and validation helpers
enforcing dimensional and range constraints.
===============================================================================
"""

import numpy as np
from typing import Tuple


def get_sample_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provide default 5x5 student performance dataset."""
    # What is used: np.array creation of students, subjects, and 2D marks matrix.
    # Why it is used: Serves as seed dataset for Student Performance Analyzer.
    # How it works: Returns 1D students array, 1D subjects array, and 2D float marks matrix.
    students = np.array(["Rahul", "Aisha", "Rohan", "Sneha", "Arjun"])
    subjects = np.array(["Python", "SQL", "Statistics", "Math", "Communication"])

    marks = np.array([
        [85.0, 90.0, 78.0, 92.0, 88.0],  # Rahul -> Avg 86.6
        [72.0, 68.0, 75.0, 80.0, 77.0],  # Aisha -> Avg 74.4
        [95.0, 91.0, 89.0, 94.0, 96.0],  # Rohan -> Avg 93.0 (Best)
        [60.0, 65.0, 70.0, 58.0, 62.0],  # Sneha -> Avg 63.0 (Lowest)
        [88.0, 84.0, 90.0, 86.0, 91.0]   # Arjun -> Avg 87.8
    ], dtype=np.float64)

    return students, subjects, marks


def validate_dataset(students: np.ndarray, subjects: np.ndarray, marks: np.ndarray) -> None:
    """Validate dimensional consistency and score boundaries for dataset."""
    # What is used: NumPy ndim, shape, and comparison validation checks.
    # Why it is used: Ensures dataset satisfies domain constraints before numerical operations.
    # How it works: Raises ValueError if marks is not 2D, or if dimensions mismatch students/subjects.
    if marks.ndim != 2:
        raise ValueError(f"Marks must be a 2D array, got ndim={marks.ndim}")

    if students.ndim != 1:
        raise ValueError(f"Students must be a 1D array, got ndim={students.ndim}")

    if subjects.ndim != 1:
        raise ValueError(f"Subjects must be a 1D array, got ndim={subjects.ndim}")

    if marks.shape[0] != students.size:
        raise ValueError(f"Student count mismatch: marks rows={marks.shape[0]}, students count={students.size}")

    if marks.shape[1] != subjects.size:
        raise ValueError(f"Subject count mismatch: marks columns={marks.shape[1]}, subjects count={subjects.size}")

    if np.any(marks < 0.0) or np.any(marks > 100.0):
        raise ValueError("Invalid score detected: All marks must fall in range [0.0, 100.0]")

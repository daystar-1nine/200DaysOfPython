"""
===============================================================================
DAY 55 — DATASET VALIDATOR MODULE
===============================================================================
This module enforces structural dimensional rules and score boundary validation
on student marks matrices, gracefully handling missing NaN entries.
===============================================================================
"""

import numpy as np


def validate_student_dataset(students: np.ndarray, subjects: np.ndarray, marks: np.ndarray) -> None:
    """Validate 2D matrix shape, dimensional consistency, and mark ranges tolerating NaNs."""
    # What is used: NumPy ndim, shape, and np.isnan masking.
    # Why it is used: Ensures dataset integrity before running advanced analytics engine.
    # How it works: Raises ValueError if marks is not 2D, or shape mismatches student/subject count.
    if marks.ndim != 2:
        raise ValueError(f"Marks matrix must be a 2D array, got ndim={marks.ndim}")

    if students.ndim != 1:
        raise ValueError(f"Students list must be a 1D array, got ndim={students.ndim}")

    if subjects.ndim != 1:
        raise ValueError(f"Subjects list must be a 1D array, got ndim={subjects.ndim}")

    if marks.shape[0] != students.size:
        raise ValueError(f"Student count mismatch: marks rows={marks.shape[0]}, students count={students.size}")

    if marks.shape[1] != subjects.size:
        raise ValueError(f"Subject count mismatch: marks columns={marks.shape[1]}, subjects count={subjects.size}")

    # Check score range boundaries excluding NaNs
    valid_marks = marks[~np.isnan(marks)]
    if valid_marks.size > 0 and (np.any(valid_marks < 0.0) or np.any(valid_marks > 100.0)):
        raise ValueError("Invalid score detected: All numerical marks must fall in range [0.0, 100.0]")

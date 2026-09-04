"""
===============================================================================
DAY 55 — REPRODUCIBLE DATA GENERATOR MODULE WITH NaN INSERTION
===============================================================================
This module generates synthetic student dataset (100 students x 5 subjects)
using reproducible random seed state and intentionally inserts missing NaN values.
===============================================================================
"""

import numpy as np
from typing import Tuple


def generate_student_dataset(
    num_students: int = 100,
    seed: int = 42,
    insert_nans: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 100 student names, 5 subjects, and 2D marks matrix with reproducible seed."""
    # What is used: np.random.default_rng(seed) and list comprehensions.
    # Why it is used: Creates large synthetic dataset with intentional missing NaN values.
    # How it works: Generates random integers (0..100), casts to float64, and inserts NaNs.
    subjects = np.array(["Python", "SQL", "Statistics", "Math", "Communication"])
    students = np.array([f"Student_{i:03d}" for i in range(1, num_students + 1)])

    rng = np.random.default_rng(seed)
    raw_marks = rng.integers(0, 101, size=(num_students, 5)).astype(np.float64)

    if insert_nans and num_students >= 25:
        # Intentionally insert NaN missing values for data science missing value testing
        raw_marks[0, 2] = np.nan   # Student_001 Statistics missing
        raw_marks[5, 1] = np.nan   # Student_006 SQL missing
        raw_marks[20, 4] = np.nan  # Student_021 Communication missing

    return students, subjects, raw_marks

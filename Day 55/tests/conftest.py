"""
===============================================================================
DAY 55 — PYTEST FIXTURES MODULE
===============================================================================
This module provides Pytest fixtures for synthetic student datasets, subjects,
marks matrices with NaNs, and temporary report path fixtures.
===============================================================================
"""

import pytest
import numpy as np
from app.generator import generate_student_dataset


@pytest.fixture
def dataset_100_students():
    """Fixture providing 100 students, 5 subjects, and marks matrix with NaNs."""
    return generate_student_dataset(num_students=100, seed=42, insert_nans=True)


@pytest.fixture
def sample_small_dataset():
    """Fixture providing a small 3x3 dataset with a NaN value for precise calculations."""
    students = np.array(["Student_A", "Student_B", "Student_C"])
    subjects = np.array(["Python", "SQL", "Math"])
    marks = np.array([
        [90.0, 80.0, np.nan],  # Avg 85.0
        [60.0, 70.0, 80.0],    # Avg 70.0
        [40.0, 50.0, 30.0],    # Avg 40.0
    ], dtype=np.float64)
    return students, subjects, marks

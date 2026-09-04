"""
===============================================================================
DAY 54 — PYTEST FIXTURES MODULE
===============================================================================
This module provides Pytest fixtures for 2D marks matrices, student arrays,
subject arrays, and temporary output path fixtures.
===============================================================================
"""

import pytest
import numpy as np


@pytest.fixture
def sample_students() -> np.ndarray:
    """Fixture providing 1D array of test student names."""
    return np.array(["Rahul", "Aisha", "Rohan", "Sneha", "Arjun"])


@pytest.fixture
def sample_subjects() -> np.ndarray:
    """Fixture providing 1D array of test subject names."""
    return np.array(["Python", "SQL", "Statistics", "Math", "Communication"])


@pytest.fixture
def sample_marks() -> np.ndarray:
    """Fixture providing 5x5 2D matrix of floating point student marks."""
    return np.array([
        [85.0, 90.0, 78.0, 92.0, 88.0],  # Rahul -> Avg 86.6
        [72.0, 68.0, 75.0, 80.0, 77.0],  # Aisha -> Avg 74.4
        [95.0, 91.0, 89.0, 94.0, 96.0],  # Rohan -> Avg 93.0 (Best)
        [60.0, 65.0, 70.0, 58.0, 62.0],  # Sneha -> Avg 63.0 (Lowest)
        [88.0, 84.0, 90.0, 86.0, 91.0]   # Arjun -> Avg 87.8
    ], dtype=np.float64)


@pytest.fixture
def invalid_marks_out_of_range() -> np.ndarray:
    """Fixture providing 2D matrix containing out-of-range marks (> 100)."""
    return np.array([
        [105.0, 90.0, 80.0],
        [70.0, 80.0, 90.0]
    ], dtype=np.float64)

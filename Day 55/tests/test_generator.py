"""
===============================================================================
DAY 55 — TEST DATA GENERATOR MODULE
===============================================================================
This test module verifies synthetic dataset generation, shape (100, 5), reproducible
random seed state, custom counts, and intentional missing NaN value insertion.
===============================================================================
"""

import numpy as np
from app.generator import generate_student_dataset


def test_generate_student_dataset_shape_and_counts(dataset_100_students):
    """Verify generated dataset shapes, student count, subject count, and ndim."""
    # What is used: generate_student_dataset fixture.
    # Why it is used: Validates expected (100, 5) shape dimensions.
    # How it works: Checks 100 students, 5 subjects, shape (100, 5), and float64 dtype.
    students, subjects, marks = dataset_100_students
    assert students.size == 100
    assert subjects.size == 5
    assert marks.shape == (100, 5)
    assert marks.ndim == 2
    assert marks.dtype == np.float64


def test_generate_student_dataset_reproducibility():
    """Verify reproducible random dataset generation using fixed seed 42."""
    # What is used: generate_student_dataset with same seed 42.
    # Why it is used: Ensures stochastic dataset generation is deterministic across test runs.
    # How it works: Generates two datasets and checks element-wise array equality (handling NaNs).
    st1, sub1, m1 = generate_student_dataset(num_students=50, seed=42)
    st2, sub2, m2 = generate_student_dataset(num_students=50, seed=42)

    assert np.array_equal(st1, st2)
    assert np.array_equal(sub1, sub2)
    assert np.array_equal(m1, m2, equal_nan=True)


def test_generate_student_dataset_nan_insertion(dataset_100_students):
    """Verify intentional NaN insertion into generated marks matrix."""
    # What is used: np.isnan() on generated marks matrix.
    # Why it is used: Ensures missing NaN values exist for data science missing value testing.
    # How it works: Checks np.isnan(marks[0, 2]) is True and total NaN count is > 0.
    _, _, marks = dataset_100_students
    assert bool(np.isnan(marks[0, 2])) is True
    assert np.sum(np.isnan(marks)) == 3


def test_generate_student_dataset_custom_count():
    """Verify custom student count generation."""
    # What is used: generate_student_dataset(num_students=30).
    # Why it is used: Validates parametric generation.
    # How it works: Checks 30 student names and 30x5 marks matrix.
    st, sub, m = generate_student_dataset(num_students=30, seed=123, insert_nans=False)
    assert st.size == 30
    assert sub.size == 5
    assert m.shape == (30, 5)
    assert np.sum(np.isnan(m)) == 0

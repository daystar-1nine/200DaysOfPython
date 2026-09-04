"""
===============================================================================
DAY 54 — TEST DATASET & VALIDATION MODULE
===============================================================================
This test module verifies array shape, ndim, size, dtype, and validation exceptions.
===============================================================================
"""

import pytest
import numpy as np
from app.data import get_sample_dataset, validate_dataset


def test_get_sample_dataset_shapes():
    """Verify sample dataset dimensions, shapes, sizes, and dtypes."""
    # What is used: get_sample_dataset fixture.
    # Why it is used: Ensures seed dataset satisfies structural 2D shape requirements.
    # How it works: Checks 5x5 shape, size=25, ndim=2, and float64 dtype.
    students, subjects, marks = get_sample_dataset()
    assert students.size == 5
    assert subjects.size == 5
    assert marks.shape == (5, 5)
    assert marks.ndim == 2
    assert marks.size == 25
    assert marks.dtype == np.float64


def test_validate_dataset_valid(sample_students, sample_subjects, sample_marks):
    """Verify valid dataset passes validation check without error."""
    # What is used: validate_dataset function.
    # Why it is used: Confirms compliant inputs pass domain validation cleanly.
    # How it works: Passes 5x5 matrix; expects no exception.
    validate_dataset(sample_students, sample_subjects, sample_marks)


def test_validate_dataset_non_2d_marks(sample_students, sample_subjects):
    """Verify ValueError raised when marks array is not 2D."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Enforces 2D matrix constraint for student marks.
    # How it works: Passes 1D marks array; asserts ValueError.
    marks_1d = np.array([80, 90, 85])
    with pytest.raises(ValueError, match="Marks must be a 2D array"):
        validate_dataset(sample_students, sample_subjects, marks_1d)


def test_validate_dataset_mismatched_students(sample_subjects, sample_marks):
    """Verify ValueError raised when student count mismatches marks matrix rows."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Validates row-to-student alignment.
    # How it works: Passes 3 student names with 5-row marks matrix; asserts ValueError.
    bad_students = np.array(["Rahul", "Aisha", "Rohan"])
    with pytest.raises(ValueError, match="Student count mismatch"):
        validate_dataset(bad_students, sample_subjects, sample_marks)


def test_validate_dataset_mismatched_subjects(sample_students, sample_marks):
    """Verify ValueError raised when subject count mismatches marks matrix columns."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Validates column-to-subject alignment.
    # How it works: Passes 3 subject names with 5-column marks matrix; asserts ValueError.
    bad_subjects = np.array(["Python", "SQL", "Stats"])
    with pytest.raises(ValueError, match="Subject count mismatch"):
        validate_dataset(sample_students, bad_subjects, sample_marks)


def test_validate_dataset_out_of_range_marks(sample_students, sample_subjects, invalid_marks_out_of_range):
    """Verify ValueError raised when marks fall outside range [0.0, 100.0]."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Prevents impossible student score values.
    # How it works: Passes matrix containing score 105.0; asserts ValueError.
    st = np.array(["Rahul", "Aisha"])
    sub = np.array(["Python", "SQL", "Stats"])
    with pytest.raises(ValueError, match="Invalid score detected"):
        validate_dataset(st, sub, invalid_marks_out_of_range)

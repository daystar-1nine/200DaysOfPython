"""
===============================================================================
DAY 55 — TEST VALIDATOR MODULE
===============================================================================
This test module verifies dataset dimensional consistency, non-2D error catching,
student/subject count mismatches, and score boundary rules handling NaNs.
===============================================================================
"""

import pytest
import numpy as np
from app.validator import validate_student_dataset


def test_validate_student_dataset_valid(sample_small_dataset):
    """Verify valid small dataset passes validation without error."""
    # What is used: validate_student_dataset with sample_small_dataset fixture.
    # Why it is used: Ensures compliant datasets pass domain validation cleanly.
    # How it works: Passes valid 3x3 dataset containing NaNs; expects no exception.
    students, subjects, marks = sample_small_dataset
    validate_student_dataset(students, subjects, marks)


def test_validate_student_dataset_non_2d_marks(sample_small_dataset):
    """Verify ValueError raised when marks array is 1D or 3D."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Enforces 2D matrix constraint.
    # How it works: Passes 1D array; asserts ValueError.
    students, subjects, _ = sample_small_dataset
    marks_1d = np.array([80, 90, 85])
    with pytest.raises(ValueError, match="Marks matrix must be a 2D array"):
        validate_student_dataset(students, subjects, marks_1d)


def test_validate_student_dataset_student_mismatch(sample_small_dataset):
    """Verify ValueError raised when student count mismatches marks matrix rows."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Validates row-to-student alignment.
    # How it works: Passes 2 student names with 3-row marks matrix; asserts ValueError.
    _, subjects, marks = sample_small_dataset
    bad_students = np.array(["Student_A", "Student_B"])
    with pytest.raises(ValueError, match="Student count mismatch"):
        validate_student_dataset(bad_students, subjects, marks)


def test_validate_student_dataset_subject_mismatch(sample_small_dataset):
    """Verify ValueError raised when subject count mismatches marks matrix columns."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Validates column-to-subject alignment.
    # How it works: Passes 2 subject names with 3-column marks matrix; asserts ValueError.
    students, _, marks = sample_small_dataset
    bad_subjects = np.array(["Python", "SQL"])
    with pytest.raises(ValueError, match="Subject count mismatch"):
        validate_student_dataset(students, bad_subjects, marks)


def test_validate_student_dataset_out_of_range(sample_small_dataset):
    """Verify ValueError raised when non-NaN scores fall outside range [0.0, 100.0]."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Prevents invalid numeric marks while ignoring NaNs.
    # How it works: Modifies element to 150.0; asserts ValueError.
    students, subjects, marks = sample_small_dataset
    bad_marks = marks.copy()
    bad_marks[0, 0] = 150.0
    with pytest.raises(ValueError, match="Invalid score detected"):
        validate_student_dataset(students, subjects, bad_marks)

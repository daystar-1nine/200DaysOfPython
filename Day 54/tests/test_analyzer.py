"""
===============================================================================
DAY 54 — TEST NUMPY ANALYZER MODULE
===============================================================================
This test module verifies numerical calculations, axis aggregations, boolean
filtering, ranking, min-max normalization, and edge case handling.
===============================================================================
"""

import pytest
import numpy as np
from app.analyzer import (
    calculate_student_totals,
    calculate_student_averages,
    calculate_subject_averages,
    get_best_student,
    get_lowest_student,
    get_best_subject,
    filter_high_performers,
    filter_low_performers,
    rank_students,
    normalize_marks_matrix,
    analyze_performance,
)


def test_calculate_student_totals(sample_marks):
    """Verify row-wise student totals calculation (axis=1)."""
    # What is used: calculate_student_totals function.
    # Why it is used: Validates row-wise sum across columns.
    # How it works: Rahul sum (85+90+78+92+88 = 433.0).
    totals = calculate_student_totals(sample_marks)
    assert totals.size == 5
    assert totals[0] == 433.0
    assert totals[2] == 465.0  # Rohan sum


def test_calculate_student_averages(sample_marks):
    """Verify row-wise student averages calculation (axis=1)."""
    # What is used: calculate_student_averages function.
    # Why it is used: Validates row-wise mean calculation.
    # How it works: Rahul avg (433 / 5 = 86.6), Rohan avg (465 / 5 = 93.0).
    averages = calculate_student_averages(sample_marks)
    assert averages.size == 5
    assert np.isclose(averages[0], 86.6)
    assert np.isclose(averages[2], 93.0)


def test_calculate_subject_averages(sample_marks):
    """Verify column-wise subject averages calculation (axis=0)."""
    # What is used: calculate_subject_averages function.
    # Why it is used: Validates column-wise mean down rows.
    # How it works: Python column avg (85+72+95+60+88) / 5 = 400 / 5 = 80.0.
    subj_avgs = calculate_subject_averages(sample_marks)
    assert subj_avgs.size == 5
    assert np.isclose(subj_avgs[0], 80.0)


def test_get_best_and_lowest_student(sample_students, sample_marks):
    """Verify identification of best and lowest scoring students using argmax and argmin."""
    # What is used: get_best_student and get_lowest_student functions.
    # Why it is used: Validates extremal student identification via argmax and argmin.
    # How it works: Rohan has max average 93.0%, Sneha has lowest average 63.0%.
    best_name, best_score = get_best_student(sample_students, sample_marks)
    low_name, low_score = get_lowest_student(sample_students, sample_marks)

    assert best_name == "Rohan"
    assert np.isclose(best_score, 93.0)
    assert low_name == "Sneha"
    assert np.isclose(low_score, 63.0)


def test_get_best_subject(sample_subjects, sample_marks):
    """Verify identification of best performing subject."""
    # What is used: get_best_subject function.
    # Why it is used: Validates subject column average maximum.
    # How it works: Python, SQL, Math, etc.; determines highest average subject.
    best_sub, best_sc = get_best_subject(sample_subjects, sample_marks)
    assert best_sub in ["Python", "SQL", "Math", "Communication"]
    assert best_sc > 80.0


def test_filter_high_and_low_performers(sample_students, sample_marks):
    """Verify boolean filtering for high and low performers."""
    # What is used: filter_high_performers and filter_low_performers.
    # Why it is used: Validates Boolean mask filtering without loops.
    # How it works: High (>80) includes Rahul, Rohan, Arjun; Low (<60) includes none (Sneha is 63.0).
    high = filter_high_performers(sample_students, sample_marks, threshold=80.0)
    low = filter_low_performers(sample_students, sample_marks, threshold=60.0)

    assert set(high) == {"Rahul", "Rohan", "Arjun"}
    assert len(low) == 0


def test_rank_students(sample_students, sample_marks):
    """Verify student ranking descending by average score using argsort."""
    # What is used: rank_students function.
    # Why it is used: Validates argsort index sorting.
    # How it works: Rank 1 is Rohan (93.0%), Rank 2 is Arjun (87.8%), Rank 3 is Rahul (86.6%).
    rankings = rank_students(sample_students, sample_marks)
    assert len(rankings) == 5
    assert rankings[0][1] == "Rohan"
    assert rankings[1][1] == "Arjun"
    assert rankings[2][1] == "Rahul"


def test_normalize_marks_matrix(sample_marks):
    """Verify min-max score normalization formula (x - min) / (max - min)."""
    # What is used: normalize_marks_matrix function.
    # Why it is used: Validates rescaling of matrix values into range [0.0, 1.0].
    # How it works: Min score in sample_marks is 58.0, Max score is 96.0.
    norm = normalize_marks_matrix(sample_marks)
    assert norm.shape == (5, 5)
    assert np.isclose(np.min(norm), 0.0)
    assert np.isclose(np.max(norm), 1.0)


def test_analyze_performance_full(sample_students, sample_subjects, sample_marks):
    """Verify end-to-end performance analysis execution."""
    # What is used: analyze_performance function.
    # Why it is used: Validates full analytics payload generation.
    # How it works: Checks overall class average, best student, and rankings.
    res = analyze_performance(sample_students, sample_subjects, sample_marks)
    assert res["student_count"] == 5
    assert res["subject_count"] == 5
    assert np.isclose(res["overall_class_average"], 80.96)
    assert res["best_student"][0] == "Rohan"


def test_empty_marks_array_edge_cases():
    """Verify defensive handling when arrays are empty."""
    # What is used: Empty NumPy array inputs.
    # Why it is used: Ensures zero-size arrays return safe default fallbacks.
    # How it works: Evaluates totals, averages, best student, and rankings on empty arrays.
    empty_marks = np.array([], dtype=np.float64).reshape(0, 0)
    empty_students = np.array([], dtype=str)
    empty_subjects = np.array([], dtype=str)

    assert calculate_student_totals(empty_marks).size == 0
    assert calculate_student_averages(empty_marks).size == 0
    assert get_best_student(empty_students, empty_marks) == ("None", 0.0)
    assert get_lowest_student(empty_students, empty_marks) == ("None", 0.0)
    assert rank_students(empty_students, empty_marks) == []

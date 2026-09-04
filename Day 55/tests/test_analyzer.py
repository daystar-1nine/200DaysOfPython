"""
===============================================================================
DAY 55 — TEST ADVANCED NUMPY ANALYZER MODULE
===============================================================================
This test module verifies NaN-aware aggregations, np.where pass/fail branching,
np.select grade distributions, np.argsort rankings, and full analytics integration.
===============================================================================
"""

import pytest
import numpy as np
from app.analyzer import (
    get_student_totals,
    get_student_averages,
    get_subject_averages,
    get_overall_class_average,
    get_class_extrema_marks,
    get_best_student_v2,
    get_lowest_student_v2,
    get_subject_extrema,
    get_top_n_students,
    compute_pass_fail_metrics,
    compute_grade_distribution,
    analyze_student_performance_v2,
)


def test_get_student_totals_nan_aware(sample_small_dataset):
    """Verify row-wise nansum calculation for student totals."""
    # What is used: get_student_totals function.
    # Why it is used: Validates row-wise nansum ignoring missing NaN.
    # How it works: Student A (90+80+NaN = 170.0), Student B (60+70+80 = 210.0).
    _, _, marks = sample_small_dataset
    totals = get_student_totals(marks)
    assert totals.size == 3
    assert totals[0] == 170.0
    assert totals[1] == 210.0


def test_get_student_averages_nan_aware(sample_small_dataset):
    """Verify row-wise nanmean calculation for student averages."""
    # What is used: get_student_averages function.
    # Why it is used: Validates row-wise nanmean calculation.
    # How it works: Student A avg ((90+80)/2 = 85.0), Student B (210/3 = 70.0), Student C (120/3 = 40.0).
    _, _, marks = sample_small_dataset
    averages = get_student_averages(marks)
    assert averages.size == 3
    assert averages[0] == 85.0
    assert averages[1] == 70.0
    assert averages[2] == 40.0


def test_get_subject_averages_nan_aware(sample_small_dataset):
    """Verify column-wise nanmean calculation for subject averages."""
    # What is used: get_subject_averages function.
    # Why it is used: Validates column-wise nanmean down rows.
    # How it works: Python avg (90+60+40)/3 = 63.33, Math avg (80+30)/2 = 55.0.
    _, _, marks = sample_small_dataset
    subj_avgs = get_subject_averages(marks)
    assert subj_avgs.size == 3
    assert np.isclose(subj_avgs[0], 63.333333333333336)
    assert np.isclose(subj_avgs[2], 55.0)


def test_get_overall_class_average_and_extrema(sample_small_dataset):
    """Verify class overall average, max, and min mark calculation ignoring NaNs."""
    # What is used: get_overall_class_average and get_class_extrema_marks functions.
    # Why it is used: Validates class-wide aggregate metrics.
    # How it works: Evaluates nanmean (520 / 8 valid entries = 65.0), nanmax (90.0), nanmin (30.0).
    _, _, marks = sample_small_dataset
    overall_avg = get_overall_class_average(marks)
    max_mark, min_mark = get_class_extrema_marks(marks)

    assert overall_avg == 62.5
    assert max_mark == 90.0
    assert min_mark == 30.0


def test_get_best_and_lowest_student_v2(sample_small_dataset):
    """Verify identification of best and lowest scoring students using nanargmax and nanargmin."""
    # What is used: get_best_student_v2 and get_lowest_student_v2.
    # Why it is used: Validates extremal student lookup via nanargmax/nanargmin.
    # How it works: Student A has highest average (85.0%), Student C has lowest average (40.0%).
    students, _, marks = sample_small_dataset
    best_name, best_score = get_best_student_v2(students, marks)
    low_name, low_score = get_lowest_student_v2(students, marks)

    assert best_name == "Student_A"
    assert best_score == 85.0
    assert low_name == "Student_C"
    assert low_score == 40.0


def test_get_subject_extrema(sample_small_dataset):
    """Verify identification of best and lowest performing subjects."""
    # What is used: get_subject_extrema function.
    # Why it is used: Validates best/worst subject determination.
    # How it works: SQL has highest avg (66.67%), Math has lowest avg (55.0%).
    _, subjects, marks = sample_small_dataset
    (best_sub, best_sc), (low_sub, low_sc) = get_subject_extrema(subjects, marks)

    assert best_sub == "SQL"
    assert low_sub == "Math"


def test_get_top_n_students(sample_small_dataset):
    """Verify top N student ranking descending by average using argsort."""
    # What is used: get_top_n_students function.
    # Why it is used: Validates argsort ranking order.
    # How it works: Rank 1 is Student_A (85.0%), Rank 2 is Student_B (70.0%), Rank 3 is Student_C (40.0%).
    students, _, marks = sample_small_dataset
    top_3 = get_top_n_students(students, marks, top_n=3)

    assert len(top_3) == 3
    assert top_3[0][1] == "Student_A"
    assert top_3[1][1] == "Student_B"
    assert top_3[2][1] == "Student_C"


def test_compute_pass_fail_metrics(sample_small_dataset):
    """Verify pass/fail classification using np.where."""
    # What is used: compute_pass_fail_metrics function.
    # Why it is used: Validates element-wise boolean condition branching with np.where.
    # How it works: All 3 students have averages >= 40.0 (85.0, 70.0, 40.0) -> 3 Passes, 0 Fails.
    _, _, marks = sample_small_dataset
    pf = compute_pass_fail_metrics(marks, pass_threshold=40.0)

    assert pf["pass_count"] == 3
    assert pf["fail_count"] == 0
    assert pf["pass_percentage"] == 100.0


def test_compute_grade_distribution(sample_small_dataset):
    """Verify multi-tier letter grade classification using np.select and np.unique counts."""
    # What is used: compute_grade_distribution function.
    # Why it is used: Validates np.select grade assignment and frequency counting.
    # How it works: Student A (85.0 -> Grade A), Student B (70.0 -> Grade B), Student C (40.0 -> Grade E).
    _, _, marks = sample_small_dataset
    grades = compute_grade_distribution(marks)

    assert grades["A"] == 1
    assert grades["B"] == 1
    assert grades["E"] == 1
    assert grades["F"] == 0


def test_analyze_student_performance_v2_full(dataset_100_students):
    """Verify full end-to-end analytics pipeline on 100-student dataset."""
    # What is used: analyze_student_performance_v2 function.
    # Why it is used: Validates complete analytics payload generation for 100 students.
    # How it works: Checks student count, subject count, top 10 rankings length, and pass/fail dict.
    students, subjects, marks = dataset_100_students
    analysis = analyze_student_performance_v2(students, subjects, marks)

    assert analysis["student_count"] == 100
    assert analysis["subject_count"] == 5
    assert len(analysis["top_10_students"]) == 10
    assert "pass_count" in analysis["pass_fail"]
    assert "A+" in analysis["grade_distribution"]


def test_empty_marks_array_edge_cases():
    """Verify defensive handling when arrays are empty in V2 analyzer."""
    # What is used: Empty NumPy array inputs.
    # Why it is used: Ensures zero-size arrays return safe default fallbacks without crashing.
    # How it works: Evaluates totals, averages, best student, and rankings on empty arrays.
    empty_marks = np.array([], dtype=np.float64).reshape(0, 0)
    empty_students = np.array([], dtype=str)
    empty_subjects = np.array([], dtype=str)

    assert get_student_totals(empty_marks).size == 0
    assert get_student_averages(empty_marks).size == 0
    assert get_overall_class_average(empty_marks) == 0.0
    assert get_best_student_v2(empty_students, empty_marks) == ("None", 0.0)
    assert get_lowest_student_v2(empty_students, empty_marks) == ("None", 0.0)
    assert get_top_n_students(empty_students, empty_marks) == []

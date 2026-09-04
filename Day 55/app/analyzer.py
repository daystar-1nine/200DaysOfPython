"""
===============================================================================
DAY 55 — ADVANCED NUMPY STUDENT ANALYTICS ENGINE MODULE
===============================================================================
This module executes advanced NaN-aware NumPy analytics, grade classification
with np.select, pass/fail branching with np.where, and ranking via np.argsort.
===============================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from app.validator import validate_student_dataset


def get_student_totals(marks: np.ndarray) -> np.ndarray:
    """Calculate row-wise total marks per student ignoring NaNs."""
    # What is used: np.nansum with axis=1.
    # Why it is used: Sums valid score entries across subject columns per student.
    # How it works: Ignores missing NaN entries and returns 1D totals array.
    if marks.size == 0:
        return np.array([], dtype=np.float64)
    return np.nansum(marks, axis=1)


def get_student_averages(marks: np.ndarray) -> np.ndarray:
    """Calculate row-wise average marks per student ignoring NaNs."""
    # What is used: np.nanmean with axis=1.
    # Why it is used: Computes mean student performance ignoring NaN missing values.
    # How it works: Returns 1D array of student averages.
    if marks.size == 0:
        return np.array([], dtype=np.float64)
    return np.nanmean(marks, axis=1)


def get_subject_averages(marks: np.ndarray) -> np.ndarray:
    """Calculate column-wise subject averages ignoring NaNs."""
    # What is used: np.nanmean with axis=0.
    # Why it is used: Computes mean score down rows for each subject column.
    # How it works: Returns 1D array of subject average scores.
    if marks.size == 0:
        return np.array([], dtype=np.float64)
    return np.nanmean(marks, axis=0)


def get_overall_class_average(marks: np.ndarray) -> float:
    """Calculate overall mean average score across entire dataset ignoring NaNs."""
    if marks.size == 0:
        return 0.0
    return float(np.nanmean(marks))


def get_class_extrema_marks(marks: np.ndarray) -> Tuple[float, float]:
    """Identify overall highest mark and lowest mark ignoring NaNs."""
    if marks.size == 0:
        return (0.0, 0.0)
    return (float(np.nanmax(marks)), float(np.nanmin(marks)))


def get_best_student_v2(students: np.ndarray, marks: np.ndarray) -> Tuple[str, float]:
    """Identify highest scoring student using np.nanargmax."""
    # What is used: np.nanargmax on student averages array.
    # Why it is used: Finds position index of maximum student average in $O(N)$ time.
    # How it works: Extracts student name at max index and average score.
    if marks.size == 0 or students.size == 0:
        return ("None", 0.0)

    averages = get_student_averages(marks)
    best_idx = np.nanargmax(averages)
    return (str(students[best_idx]), float(averages[best_idx]))


def get_lowest_student_v2(students: np.ndarray, marks: np.ndarray) -> Tuple[str, float]:
    """Identify lowest scoring student using np.nanargmin."""
    # What is used: np.nanargmin on student averages array.
    # Why it is used: Finds position index of minimum student average.
    # How it works: Extracts student name at min index and average score.
    if marks.size == 0 or students.size == 0:
        return ("None", 0.0)

    averages = get_student_averages(marks)
    lowest_idx = np.nanargmin(averages)
    return (str(students[lowest_idx]), float(averages[lowest_idx]))


def get_subject_extrema(subjects: np.ndarray, marks: np.ndarray) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    """Identify best subject and lowest subject using np.nanargmax and np.nanargmin."""
    if marks.size == 0 or subjects.size == 0:
        return (("None", 0.0), ("None", 0.0))

    subj_avgs = get_subject_averages(marks)
    best_idx = np.nanargmax(subj_avgs)
    lowest_idx = np.nanargmin(subj_avgs)
    return (
        (str(subjects[best_idx]), float(subj_avgs[best_idx])),
        (str(subjects[lowest_idx]), float(subj_avgs[lowest_idx])),
    )


def get_top_n_students(students: np.ndarray, marks: np.ndarray, top_n: int = 10) -> List[Tuple[int, str, float]]:
    """Rank top N students descending by average score using np.argsort."""
    # What is used: np.argsort(averages)[::-1] for descending index sorting.
    # Why it is used: Orders top students from highest to lowest performer cleanly.
    # How it works: Iterates sorted index positions up to top_n and constructs rank tuples.
    if marks.size == 0 or students.size == 0:
        return []

    averages = get_student_averages(marks)
    sorted_indices = np.argsort(averages)[::-1][:top_n]

    top_list = []
    for rank_idx, idx in enumerate(sorted_indices, start=1):
        top_list.append((rank_idx, str(students[idx]), float(averages[idx])))

    return top_list


def compute_pass_fail_metrics(marks: np.ndarray, pass_threshold: float = 40.0) -> Dict[str, Any]:
    """Classify student pass/fail status using np.where and compute pass percentage."""
    # What is used: np.where(averages >= pass_threshold, "Pass", "Fail").
    # Why it is used: Executes element-wise conditional branching across all student averages.
    # How it works: Counts "Pass" and "Fail" entries, computing pass/fail percentages.
    averages = get_student_averages(marks)
    status = np.where(averages >= pass_threshold, "Pass", "Fail")

    pass_count = int(np.sum(status == "Pass"))
    fail_count = int(np.sum(status == "Fail"))
    total_students = len(averages)
    pass_pct = (pass_count / total_students * 100.0) if total_students > 0 else 0.0
    fail_pct = (fail_count / total_students * 100.0) if total_students > 0 else 0.0

    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_percentage": pass_pct,
        "fail_percentage": fail_pct,
    }


def compute_grade_distribution(marks: np.ndarray) -> Dict[str, int]:
    """Classify student grades using np.select and tally frequency distribution with np.unique."""
    # What is used: np.select(conditions, choices, default="F") and np.unique(return_counts=True).
    # Why it is used: Assigns multi-tier academic letter grades and tallies grade counts.
    # How it works: Evaluates 6 grade boundary conditions and zips unique grade counts.
    averages = get_student_averages(marks)

    conditions = [
        averages >= 90.0,
        averages >= 80.0,
        averages >= 70.0,
        averages >= 60.0,
        averages >= 50.0,
        averages >= 40.0,
    ]
    choices = ["A+", "A", "B", "C", "D", "E"]

    grades = np.select(conditions, choices, default="F")

    unique_grades, counts = np.unique(grades, return_counts=True)
    distribution_dict = dict(zip(unique_grades, counts))

    # Ensure all standard grade categories exist in returned dict
    all_grades = ["A+", "A", "B", "C", "D", "E", "F"]
    return {g: int(distribution_dict.get(g, 0)) for g in all_grades}


def analyze_student_performance_v2(
    students: np.ndarray, subjects: np.ndarray, marks: np.ndarray
) -> Dict[str, Any]:
    """Execute end-to-end advanced analytics pipeline on student dataset."""
    # What is used: Modular NumPy analysis functions.
    # Why it is used: Ingests student dataset, validates dimensions, and computes complete analytics.
    # How it works: Calls validation, calculates averages, rankings, pass/fail, and grade distribution.
    validate_student_dataset(students, subjects, marks)

    tot = get_student_totals(marks)
    avg = get_student_averages(marks)
    subj_avg = get_subject_averages(marks)
    overall_avg = get_overall_class_average(marks)
    max_mark, min_mark = get_class_extrema_marks(marks)
    best_st, best_st_score = get_best_student_v2(students, marks)
    low_st, low_st_score = get_lowest_student_v2(students, marks)
    (best_sub, best_sub_score), (low_sub, low_sub_score) = get_subject_extrema(subjects, marks)
    top_10 = get_top_n_students(students, marks, top_n=10)
    pass_fail = compute_pass_fail_metrics(marks, pass_threshold=40.0)
    grade_dist = compute_grade_distribution(marks)

    high_performers_mask = (avg >= 80.0)
    low_performers_mask = (avg < 40.0)

    return {
        "student_count": students.size,
        "subject_count": subjects.size,
        "overall_class_average": overall_avg,
        "highest_mark": max_mark,
        "lowest_mark": min_mark,
        "student_totals": tot,
        "student_averages": avg,
        "subject_averages": dict(zip(subjects, subj_avg)),
        "best_student": (best_st, best_st_score),
        "lowest_student": (low_st, low_st_score),
        "best_subject": (best_sub, best_sub_score),
        "lowest_subject": (low_sub, low_sub_score),
        "high_performers_count": int(np.sum(high_performers_mask)),
        "low_performers_count": int(np.sum(low_performers_mask)),
        "top_10_students": top_10,
        "pass_fail": pass_fail,
        "grade_distribution": grade_dist,
    }

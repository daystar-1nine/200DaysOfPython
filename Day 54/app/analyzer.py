"""
===============================================================================
DAY 54 — NUMPY STUDENT ANALYZER MODULE
===============================================================================
This module executes vector calculations for student performance metrics using NumPy.
===============================================================================
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from app.data import validate_dataset


def calculate_student_totals(marks: np.ndarray) -> np.ndarray:
    """Calculate total marks per student across all subjects (row-wise)."""
    # What is used: np.sum with axis=1.
    # Why it is used: Sums elements across columns for each individual student row.
    # How it works: Returns 1D array of total marks.
    if marks.size == 0:
        return np.array([], dtype=np.float64)
    return np.sum(marks, axis=1)


def calculate_student_averages(marks: np.ndarray) -> np.ndarray:
    """Calculate average marks per student across all subjects (row-wise)."""
    # What is used: np.mean with axis=1.
    # Why it is used: Computes arithmetic mean for each student.
    # How it works: Returns 1D array of student averages.
    if marks.size == 0:
        return np.array([], dtype=np.float64)
    return np.mean(marks, axis=1)


def calculate_subject_averages(marks: np.ndarray) -> np.ndarray:
    """Calculate average marks per subject across all students (column-wise)."""
    # What is used: np.mean with axis=0.
    # Why it is used: Computes mean score down rows for each subject column.
    # How it works: Returns 1D array of subject average scores.
    if marks.size == 0:
        return np.array([], dtype=np.float64)
    return np.mean(marks, axis=0)


def get_best_student(students: np.ndarray, marks: np.ndarray) -> Tuple[str, float]:
    """Identify top scoring student name and average mark using np.argmax."""
    # What is used: np.argmax on student_averages array.
    # Why it is used: Finds position index of maximum student average in $O(N)$ time.
    # How it works: Extracts student name at max index and average score.
    if marks.size == 0 or students.size == 0:
        return ("None", 0.0)

    averages = calculate_student_averages(marks)
    best_idx = np.argmax(averages)
    return (str(students[best_idx]), float(averages[best_idx]))


def get_lowest_student(students: np.ndarray, marks: np.ndarray) -> Tuple[str, float]:
    """Identify lowest scoring student name and average mark using np.argmin."""
    # What is used: np.argmin on student_averages array.
    # Why it is used: Finds position index of minimum student average.
    # How it works: Extracts student name at min index and average score.
    if marks.size == 0 or students.size == 0:
        return ("None", 0.0)

    averages = calculate_student_averages(marks)
    lowest_idx = np.argmin(averages)
    return (str(students[lowest_idx]), float(averages[lowest_idx]))


def get_best_subject(subjects: np.ndarray, marks: np.ndarray) -> Tuple[str, float]:
    """Identify highest performing subject name and average score."""
    # What is used: np.argmax on subject_averages array.
    # Why it is used: Determines subject with highest overall mean score.
    # How it works: Extracts subject name at max index and column average.
    if marks.size == 0 or subjects.size == 0:
        return ("None", 0.0)

    subj_avgs = calculate_subject_averages(marks)
    best_idx = np.argmax(subj_avgs)
    return (str(subjects[best_idx]), float(subj_avgs[best_idx]))


def filter_high_performers(students: np.ndarray, marks: np.ndarray, threshold: float = 80.0) -> np.ndarray:
    """Filter student names with average score >= threshold using Boolean masking."""
    # What is used: Vectorized boolean mask (averages >= threshold).
    # Why it is used: Filters high performing students without Python for-loops.
    # How it works: Indexes students array with boolean mask.
    if marks.size == 0 or students.size == 0:
        return np.array([], dtype=str)

    averages = calculate_student_averages(marks)
    mask = (averages >= threshold)
    return students[mask]


def filter_low_performers(students: np.ndarray, marks: np.ndarray, threshold: float = 60.0) -> np.ndarray:
    """Filter student names with average score < threshold using Boolean masking."""
    # What is used: Vectorized boolean mask (averages < threshold).
    # Why it is used: Identifies students requiring academic assistance.
    # How it works: Indexes students array with boolean mask.
    if marks.size == 0 or students.size == 0:
        return np.array([], dtype=str)

    averages = calculate_student_averages(marks)
    mask = (averages < threshold)
    return students[mask]


def rank_students(students: np.ndarray, marks: np.ndarray) -> List[Tuple[int, str, float]]:
    """Rank students descending by average score using np.argsort."""
    # What is used: np.argsort(averages)[::-1] for descending index sorting.
    # Why it is used: Orders students from highest to lowest performer cleanly.
    # How it works: Iterates sorted index positions and constructs rank tuples.
    if marks.size == 0 or students.size == 0:
        return []

    averages = calculate_student_averages(marks)
    sorted_indices = np.argsort(averages)[::-1]

    rankings = []
    for rank_idx, idx in enumerate(sorted_indices, start=1):
        rankings.append((rank_idx, str(students[idx]), float(averages[idx])))

    return rankings


def normalize_marks_matrix(marks: np.ndarray) -> np.ndarray:
    """Normalize marks matrix to range [0.0, 1.0] using Min-Max scaling formula."""
    # What is used: Vectorized array math (marks - min) / (max - min).
    # Why it is used: Scales marks to uniform range [0.0, 1.0].
    # How it works: Evaluates min and max across entire matrix and rescales.
    if marks.size == 0:
        return marks.astype(np.float64)

    min_val = np.min(marks)
    max_val = np.max(marks)

    if max_val == min_val:
        return np.zeros_like(marks, dtype=np.float64)

    return (marks - min_val) / (max_val - min_val)


def analyze_performance(students: np.ndarray, subjects: np.ndarray, marks: np.ndarray) -> Dict[str, Any]:
    """Execute complete numerical analysis on student performance dataset."""
    # What is used: Full analytics pipeline orchestration.
    # Why it is used: Validates dataset and aggregates all performance metrics.
    # How it works: Calls validate_dataset and combines analytical results into dictionary.
    validate_dataset(students, subjects, marks)

    tot = calculate_student_totals(marks)
    avg = calculate_student_averages(marks)
    subj_avg = calculate_subject_averages(marks)
    best_st, best_st_score = get_best_student(students, marks)
    low_st, low_st_score = get_lowest_student(students, marks)
    best_sub, best_sub_score = get_best_subject(subjects, marks)
    high_perf = filter_high_performers(students, marks, threshold=80.0)
    low_perf = filter_low_performers(students, marks, threshold=60.0)
    rankings = rank_students(students, marks)
    overall_avg = float(np.mean(marks))

    return {
        "student_count": students.size,
        "subject_count": subjects.size,
        "overall_class_average": overall_avg,
        "student_totals": tot,
        "student_averages": avg,
        "subject_averages": dict(zip(subjects, subj_avg)),
        "best_student": (best_st, best_st_score),
        "lowest_student": (low_st, low_st_score),
        "best_subject": (best_sub, best_sub_score),
        "high_performers": high_perf,
        "low_performers": low_perf,
        "rankings": rankings,
        "normalized_marks": normalize_marks_matrix(marks),
    }

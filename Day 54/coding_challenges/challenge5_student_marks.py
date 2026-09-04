"""
===============================================================================
DAY 54 — CODING CHALLENGE 5: 2D STUDENT MARKS MATRIX ANALYZER
===============================================================================
Topic: 2D Aggregations, argmax, argmin, and Boolean Masking
Goal: Given a 4x3 student marks matrix, compute student & subject averages,
      identify best/worst performers and subjects.
===============================================================================
"""

import numpy as np


def analyze_student_marks_matrix(marks: np.ndarray):
    """Analyze 2D student marks matrix."""
    # What is used: Axis aggregations (axis=1 for student, axis=0 for subject), argmax, argmin.
    # Why it is used: Solves 2D matrix student performance analytics cleanly.
    # How it works: Computes mean across axes and extracts indices of extrema.
    student_averages = np.mean(marks, axis=1)
    subject_averages = np.mean(marks, axis=0)

    best_student_idx = np.argmax(student_averages)
    worst_student_idx = np.argmin(student_averages)
    best_subject_idx = np.argmax(subject_averages)

    high_performers_count = np.sum(student_averages > 80.0)

    return {
        "student_averages": student_averages,
        "subject_averages": subject_averages,
        "best_student_idx": best_student_idx,
        "worst_student_idx": worst_student_idx,
        "best_subject_idx": best_subject_idx,
        "high_performers_count": high_performers_count,
    }


if __name__ == "__main__":
    marks_matrix = np.array([
        [80, 90, 85],  # Student 0 -> Avg 85.0
        [70, 75, 80],  # Student 1 -> Avg 75.0
        [95, 92, 98],  # Student 2 -> Avg 95.0 (Best)
        [60, 65, 70]   # Student 3 -> Avg 65.0 (Worst)
    ])

    res = analyze_student_marks_matrix(marks_matrix)
    print("Student Marks Analysis:")
    print("  Student Averages:", res["student_averages"])
    print("  Subject Averages:", res["subject_averages"])
    print("  Best Student Index: ", res["best_student_idx"])
    print("  Worst Student Index:", res["worst_student_idx"])
    print("  Best Subject Index: ", res["best_subject_idx"])
    print("  High Performers (>80):", res["high_performers_count"])

    assert res["best_student_idx"] == 2, "Best student index failed"
    assert res["worst_student_idx"] == 3, "Worst student index failed"
    assert res["high_performers_count"] == 2, "High performers count failed (Student 0 & 2)"
    print("[OK] Challenge 5 Passed Successfully!")

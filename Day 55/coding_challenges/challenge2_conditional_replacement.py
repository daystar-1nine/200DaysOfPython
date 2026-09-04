"""
===============================================================================
DAY 55 — CODING CHALLENGE 2: CONDITIONAL REPLACEMENT
===============================================================================
Topic: In-Place Conditional Array Mask Assignment
Goal: Given marks [35, 45, 78, 29, 90, 55], replace all marks < 40 with 40.
===============================================================================
"""

import numpy as np


def replace_failing_marks(marks: np.ndarray) -> np.ndarray:
    """Replace all elements < 40 with grace mark 40 in-place."""
    # What is used: Vectorized boolean mask condition marks < 40.
    # Why it is used: Implements thresholding / imputation without loops.
    # How it works: Assigns value 40 directly to array elements matching condition.
    result = marks.copy()
    result[result < 40] = 40
    return result


if __name__ == "__main__":
    raw_marks = np.array([35, 45, 78, 29, 90, 55])
    cleaned = replace_failing_marks(raw_marks)
    print("Raw Marks:    ", raw_marks)
    print("Cleaned Marks:", cleaned)

    expected = np.array([40, 45, 78, 40, 90, 55])
    assert np.array_equal(cleaned, expected), f"Expected {expected}, got {cleaned}"
    print("[OK] Challenge 2 Passed Successfully!")

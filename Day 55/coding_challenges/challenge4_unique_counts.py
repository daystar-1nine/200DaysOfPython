"""
===============================================================================
DAY 55 — CODING CHALLENGE 4: UNIQUE VALUE FREQUENCY COUNTS
===============================================================================
Topic: Categorical Analysis with np.unique(return_counts=True)
Goal: Given courses ["DS", "CS", "DS", "AI", "CS", "DS", "AI"], tally unique counts.
===============================================================================
"""

import numpy as np
from typing import Dict


def tally_unique_courses(courses: np.ndarray) -> Dict[str, int]:
    """Extract unique course names and their frequency counts."""
    # What is used: np.unique(courses, return_counts=True).
    # Why it is used: Computes distinct values and occurrences in single C pass.
    # How it works: Returns unique elements array and counts array, zipped into dict.
    vals, counts = np.unique(courses, return_counts=True)
    return dict(zip(vals, counts))


if __name__ == "__main__":
    courses = np.array(["DS", "CS", "DS", "AI", "CS", "DS", "AI"])
    tally = tally_unique_courses(courses)
    print("Unique Course Counts:", tally)

    assert tally["DS"] == 3, "DS count failed"
    assert tally["CS"] == 2, "CS count failed"
    assert tally["AI"] == 2, "AI count failed"
    print("[OK] Challenge 4 Passed Successfully!")

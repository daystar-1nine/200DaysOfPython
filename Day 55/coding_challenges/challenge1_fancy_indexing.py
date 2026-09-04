"""
===============================================================================
DAY 55 — CODING CHALLENGE 1: FANCY INDEXING
===============================================================================
Topic: Integer-Array / Fancy Indexing in NumPy
Goal: Given numbers 1 to 10, extract 1, 3, 5, 7, 9 using NumPy fancy indexing.
===============================================================================
"""

import numpy as np


def extract_odd_position_elements() -> np.ndarray:
    """Extract 1st, 3rd, 5th, 7th, and 9th elements from numbers 1..10."""
    # What is used: NumPy integer array fancy indexing arr[[indices]].
    # Why it is used: Selects non-contiguous array elements without loops.
    # How it works: Passes index list [0, 2, 4, 6, 8] into numbers array.
    numbers = np.arange(1, 11)
    indices = [0, 2, 4, 6, 8]
    selected = numbers[indices]
    return selected


if __name__ == "__main__":
    result = extract_odd_position_elements()
    print("Fancy Indexing Result:", result)

    expected = np.array([1, 3, 5, 7, 9])
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print("[OK] Challenge 1 Passed Successfully!")

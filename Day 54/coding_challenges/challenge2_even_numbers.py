"""
===============================================================================
DAY 54 — CODING CHALLENGE 2: EVEN NUMBERS WITH BOOLEAN INDEXING
===============================================================================
Topic: Vectorized Boolean Indexing without Python For Loops
Goal: Extract all even numbers from 1 to 100 using a Boolean mask.
===============================================================================
"""

import numpy as np


def extract_even_numbers() -> np.ndarray:
    """Extract even numbers from 1 to 100 using Boolean mask vectorization."""
    # What is used: Vectorized modulo operation % and Boolean masking.
    # Why it is used: Demonstrates loop-less filtering in $O(N)$ C execution.
    # How it works: Creates boolean mask (numbers % 2 == 0) and indexes array.
    numbers = np.arange(1, 101)
    even_mask = (numbers % 2 == 0)
    even_numbers = numbers[even_mask]
    return even_numbers


if __name__ == "__main__":
    evens = extract_even_numbers()
    print(f"Extracted {evens.size} Even Numbers:")
    print(evens)

    assert evens.size == 50, "Even count failed"
    assert evens[0] == 2, "First even number failed"
    assert evens[-1] == 100, "Last even number failed"
    assert np.all(evens % 2 == 0), "All elements must be even"
    print("[OK] Challenge 2 Passed Successfully!")

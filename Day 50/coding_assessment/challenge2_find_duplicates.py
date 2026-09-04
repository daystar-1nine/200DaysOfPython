"""
===============================================================================
DAY 50 — CODING CHALLENGE 2: FIND DUPLICATE NUMBERS IN A LIST
===============================================================================
This module identifies all numbers that appear more than once in a given list,
returning them in order of their first duplicate discovery.
===============================================================================
"""

from typing import List

def find_duplicates(numbers: List[int]) -> List[int]:
    """Find and return duplicate numbers from a list in order of occurrence."""
    # What is used: Set tracking collections (seen and duplicates).
    # Why it is used: Provides O(1) membership checks to identify repeated elements.
    # How it works: Adds elements to seen; if already present, adds to duplicates list.
    seen = set()
    duplicates = []
    for num in numbers:
        if num in seen:
            if num not in duplicates:
                duplicates.append(num)
        else:
            seen.add(num)
    return duplicates


if __name__ == "__main__":
    sample_input = [1, 2, 3, 2, 4, 5, 3]
    result = find_duplicates(sample_input)
    print(f"Input: {sample_input} -> Duplicates: {result}")
    assert result == [2, 3], "Challenge 2 Failed!"
    print("[OK] Challenge 2 Passed!")

"""
===============================================================================
DAY 53 — CODING CHALLENGE 1: CLEAN NAMES
===============================================================================
Topic: String Normalization, Trimming, and Casing
Goal: Take a list of raw dirty name strings containing leading/trailing spaces
      and inconsistent casing, and return a clean, title-cased list.
===============================================================================
"""

from typing import List


def clean_names(raw_names: List[str]) -> List[str]:
    """Clean and normalize a list of raw name strings."""
    # What is used: List comprehension with str.strip() and str.title().
    # Why it is used: Removes extraneous whitespace and standardizes name casing.
    # How it works: Iterates through each string, strips padding, and applies Title Case.
    cleaned = []
    for name in raw_names:
        if name and isinstance(name, str):
            clean_str = name.strip().title()
            if clean_str:
                cleaned.append(clean_str)
    return cleaned


if __name__ == "__main__":
    # Test case execution
    raw = [" rahul ", "AISHa", " rohan", "SNEHA "]
    result = clean_names(raw)
    print("Raw Input Names:", raw)
    print("Cleaned Names:  ", result)
    expected = ["Rahul", "Aisha", "Rohan", "Sneha"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("[OK] Challenge 1 Passed Successfully!")

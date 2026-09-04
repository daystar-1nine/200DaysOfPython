"""
===============================================================================
DAY 50 — CODING CHALLENGE 1: REVERSE A STRING WITHOUT SLICING [::-1]
===============================================================================
This module provides a function to reverse a string without using Python's
built-in extended slice operator [::-1].
===============================================================================
"""

def reverse_string(s: str) -> str:
    """Reverse a string using loop iteration without using slice [::-1]."""
    # What is used: Loop string accumulation.
    # Why it is used: Reverses character sequence manually step-by-step.
    # How it works: Prepends each character char to reversed_acc string.
    reversed_acc = ""
    for char in s:
        reversed_acc = char + reversed_acc
    return reversed_acc


if __name__ == "__main__":
    test_str = "python"
    result = reverse_string(test_str)
    print(f"Original: {test_str} -> Reversed: {result}")
    assert result == "nohtyp", "Challenge 1 Failed!"
    print("[OK] Challenge 1 Passed!")

"""
===============================================================================
DAY 50 — CODING CHALLENGE 3: WORD FREQUENCY DICTIONARY COUNTER
===============================================================================
This module processes a sentence string and returns a dictionary mapping each
unique word to its occurrence count.
===============================================================================
"""

from typing import Dict

def count_word_frequency(text: str) -> Dict[str, int]:
    """Calculate the frequency of each word in a given text string."""
    # What is used: String split method and dict accumulator.
    # Why it is used: Tokenizes string into words and tallies frequencies.
    # How it works: Splits by whitespace and updates word counts in frequency dict.
    words = text.lower().split()
    frequency: Dict[str, int] = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency


if __name__ == "__main__":
    sample_text = "python is easy and python is powerful"
    result = count_word_frequency(sample_text)
    expected = {
        "python": 2,
        "is": 2,
        "easy": 1,
        "and": 1,
        "powerful": 1,
    }
    print(f"Text: '{sample_text}'\nFrequency: {result}")
    assert result == expected, "Challenge 3 Failed!"
    print("✅ Challenge 3 Passed!")

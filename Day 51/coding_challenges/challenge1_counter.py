"""
===============================================================================
DAY 51 — CODING CHALLENGE 1: WORD FREQUENCY COUNTER
===============================================================================
This module uses collections.Counter to calculate word frequency counts from
a list of strings.
===============================================================================
"""

from collections import Counter
from typing import List, Dict


def get_word_frequencies(words: List[str]) -> Dict[str, int]:
    """Calculate frequency of each word in a list using collections.Counter."""
    # What is used: collections.Counter class.
    # Why it is used: Efficiently counts occurrences of items in an iterable.
    # How it works: Instantiates Counter object passing the list of words.
    counter = Counter(words)
    return dict(counter)


if __name__ == "__main__":
    sample_words = ["python", "sql", "python", "fastapi", "sql", "python"]
    freq = get_word_frequencies(sample_words)
    print("Word Frequencies:", freq)
    assert freq["python"] == 3
    assert freq["sql"] == 2
    assert freq["fastapi"] == 1
    print("[OK] Challenge 1 Passed!")

"""
===============================================================================
DAY 55 — CODING CHALLENGE 3: ARGSORT RANKING
===============================================================================
Topic: Array Sorting and Ranking via np.argsort
Goal: Given scores [80, 95, 72, 88, 91], rank items in descending order of score.
===============================================================================
"""

import numpy as np


def rank_scores_descending(scores: np.ndarray) -> np.ndarray:
    """Return indices that rank scores in descending order."""
    # What is used: np.argsort(scores)[::-1].
    # Why it is used: Produces rank-ordered index positions.
    # How it works: Obtains ascending indices via argsort and reverses with slice [::-1].
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices


if __name__ == "__main__":
    scores = np.array([80, 95, 72, 88, 91])
    ranked_indices = rank_scores_descending(scores)
    print("Raw Scores:        ", scores)
    print("Ranked Indices:    ", ranked_indices)
    print("Ranked Scores List:", scores[ranked_indices])

    expected_indices = np.array([1, 4, 3, 0, 2])  # 95 (idx 1), 91 (idx 4), 88 (idx 3), 80 (idx 0), 72 (idx 2)
    assert np.array_equal(ranked_indices, expected_indices), f"Expected {expected_indices}, got {ranked_indices}"
    print("[OK] Challenge 3 Passed Successfully!")

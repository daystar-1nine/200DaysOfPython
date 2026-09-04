"""
===============================================================================
DAY 54 — CODING CHALLENGE 6: MIN-MAX FEATURE NORMALIZATION
===============================================================================
Topic: Feature Scaling Formula Implementation without External Library Functions
Goal: Rescale scores [10, 20, 30, 40, 50] to range [0.0, 1.0] using (x - min)/(max - min).
===============================================================================
"""

import numpy as np


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Rescale a numeric NumPy array to range [0.0, 1.0] using Min-Max scaling formula."""
    # What is used: Vectorized array arithmetic and scalar np.min() / np.max().
    # Why it is used: Demonstrates feature scaling algorithms foundational to Machine Learning.
    # How it works: Evaluates (x - x_min) / (x_max - x_min) using broadcasting.
    if scores.size == 0:
        return scores.astype(np.float64)

    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val == min_val:
        return np.zeros_like(scores, dtype=np.float64)

    normalized = (scores - min_val) / (max_val - min_val)
    return normalized


if __name__ == "__main__":
    scores = np.array([10, 20, 30, 40, 50])
    norm = min_max_normalize(scores)
    print("Raw Input Scores:   ", scores)
    print("Normalized Output:  ", norm)

    expected = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    assert np.allclose(norm, expected), "Normalization output mismatched"
    assert norm[0] == 0.0, "Minimum normalized value must be 0.0"
    assert norm[-1] == 1.0, "Maximum normalized value must be 1.0"
    print("[OK] Challenge 6 Passed Successfully!")

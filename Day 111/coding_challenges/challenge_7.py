"""
Challenge 7: Implement Decision Threshold Tuner to maximize F1-score across probability predictions.
"""
from typing import Tuple, List
import numpy as np


def find_optimal_threshold(
    y_true: List[int],
    y_probs: List[float],
    thresholds: List[float] = None
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes F1 score.
    Returns:
        (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_probs, dtype=float)

    best_thresh = 0.5
    best_f1 = -1.0

    for t in thresholds:
        preds = (y_p >= t).astype(int)
        tp = np.sum((y_t == 1) & (preds == 1))
        fp = np.sum((y_t == 0) & (preds == 1))
        fn = np.sum((y_t == 1) & (preds == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


if __name__ == "__main__":
    y_true = [0, 0, 0, 1, 1, 1, 1, 1]
    y_probs = [0.1, 0.2, 0.6, 0.55, 0.7, 0.8, 0.85, 0.9]

    best_t, best_f1 = find_optimal_threshold(y_true, y_probs)
    print(f"Optimal Threshold: {best_t:.2f} | Best F1: {best_f1:.4f}")
    assert 0.0 < best_t < 1.0
    assert best_f1 > 0.8
    print("Challenge 7 passed successfully!")

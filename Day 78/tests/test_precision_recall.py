import numpy as np
from app.evaluation.precision_recall import compute_pr

def test_compute_pr():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9])
    pr = compute_pr(y_true, y_prob)
    assert 0.0 <= pr["average_precision"] <= 1.0
    assert len(pr["precision"]) == len(pr["recall"])

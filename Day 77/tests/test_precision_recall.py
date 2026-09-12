import numpy as np
from app.evaluation.precision_recall import compute_binary_pr, compute_multiclass_pr

def test_binary_pr():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9])
    pr = compute_binary_pr(y_true, y_prob)
    assert 0.0 <= pr["average_precision"] <= 1.0
    assert len(pr["precision"]) == len(pr["recall"])

def test_multiclass_pr():
    classes = ["Low", "Medium", "High"]
    y_true = np.array(["Low", "Medium", "High", "Low"])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05]
    ])
    pr = compute_multiclass_pr(y_true, y_prob, classes=classes)
    assert "macro_ap" in pr
    assert 0.0 <= pr["macro_ap"] <= 1.0

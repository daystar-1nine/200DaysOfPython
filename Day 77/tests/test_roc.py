import numpy as np
from app.evaluation.roc import compute_binary_roc, compute_multiclass_roc

def test_binary_roc():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    roc = compute_binary_roc(y_true, y_prob)
    assert roc["auc"] == 1.0
    assert 0.0 <= roc["optimal_threshold_youden"] <= 1.0

def test_multiclass_roc():
    classes = ["Low", "Medium", "High"]
    y_true = np.array(["Low", "Medium", "High", "Low"])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05]
    ])
    roc = compute_multiclass_roc(y_true, y_prob, classes=classes)
    assert "macro_auc" in roc
    assert set(roc["roc_curves"].keys()) == set(classes)

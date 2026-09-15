import numpy as np
from app.evaluation.roc import compute_roc

def test_compute_roc():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    roc = compute_roc(y_true, y_prob)
    assert roc["auc"] == 1.0
    assert 0.0 <= roc["optimal_threshold_youden"] <= 1.0
    assert len(roc["fpr"]) == len(roc["tpr"])

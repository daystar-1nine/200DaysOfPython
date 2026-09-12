import numpy as np
from app.evaluation.metrics import calculate_binary_metrics, calculate_multiclass_metrics

def test_binary_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    m = calculate_binary_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0

def test_multiclass_metrics_keys():
    y_true = np.array(["Low", "Medium", "High", "Low"])
    y_pred = np.array(["Low", "Medium", "High", "High"])
    m = calculate_multiclass_metrics(y_true, y_pred, labels=["Low", "Medium", "High"])
    assert "accuracy" in m
    assert "macro_f1" in m
    assert "weighted_f1" in m
    assert "per_class" in m
    assert set(m["per_class"].keys()) == {"Low", "Medium", "High"}

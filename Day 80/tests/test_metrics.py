import numpy as np
from app.evaluation.metrics import evaluate_classifier_metrics
from app.evaluation.confusion_matrix import compute_confusion_breakdown

def test_metrics_values():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])
    m = evaluate_classifier_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= m['accuracy'] <= 1.0
    assert 0.0 <= m['precision'] <= 1.0
    assert 0.0 <= m['recall'] <= 1.0
    assert 0.0 <= m['f1'] <= 1.0

def test_specificity():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    m = evaluate_classifier_metrics(y_true, y_pred)
    assert m['specificity'] == 0.75

def test_balanced_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = evaluate_classifier_metrics(y_true, y_pred)
    assert m['balanced_accuracy'] == 1.0

def test_roc_auc_and_ap():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate_classifier_metrics(y_true, y_true, y_prob)
    assert m['roc_auc'] == 1.0
    assert m['average_precision'] == 1.0

def test_brier_score():
    y_true = np.array([1, 0])
    y_prob = np.array([1.0, 0.0])
    m = evaluate_classifier_metrics(y_true, y_true, y_prob)
    assert m['brier_score'] == 0.0

def test_confusion_matrix_breakdown():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    res = compute_confusion_breakdown(y_true, y_pred)
    assert res['TP'] == 1
    assert res['FP'] == 1
    assert res['FN'] == 1
    assert res['TN'] == 1
    assert res['Total'] == 4

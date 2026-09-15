import numpy as np
from app.evaluation.metrics import calculate_classification_metrics
from app.evaluation.confusion_matrix import compute_confusion_matrix_breakdown
from app.evaluation.roc import compute_roc_curve_data
from app.evaluation.precision_recall import compute_pr_curve_data

def test_calculate_classification_metrics_basic():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])
    m = calculate_classification_metrics(y_true, y_pred, y_prob)
    assert 'accuracy' in m and 'precision' in m and 'recall' in m and 'f1_score' in m
    assert 'roc_auc' in m and 'pr_auc' in m
    assert 0.0 <= m['accuracy'] <= 1.0

def test_specificity_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    m = calculate_classification_metrics(y_true, y_pred)
    assert m['specificity'] == 0.75

def test_confusion_matrix_breakdown():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    cm = compute_confusion_matrix_breakdown(y_true, y_pred)
    assert cm['TP'] == 1
    assert cm['FP'] == 1
    assert cm['FN'] == 1
    assert cm['TN'] == 1
    assert cm['total'] == 4

def test_roc_curve_data():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, th, roc_auc = compute_roc_curve_data(y_true, y_prob)
    assert len(fpr) == len(tpr)
    assert 0.0 <= roc_auc <= 1.0

def test_precision_recall_curve_data():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    prec, rec, th, pr_auc = compute_pr_curve_data(y_true, y_prob)
    assert len(prec) == len(rec)
    assert 0.0 <= pr_auc <= 1.0

def test_brier_and_log_loss():
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])
    m = calculate_classification_metrics(y_true, y_true, y_prob)
    assert 'brier_score' in m
    assert m['brier_score'] < 0.05

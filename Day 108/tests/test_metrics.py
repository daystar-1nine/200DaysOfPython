import pytest
import numpy as np
from app.evaluation.metrics import compute_classification_metrics
from app.evaluation.confusion_matrix import compute_confusion_matrix_details
from app.evaluation.roc import compute_roc_curve_data
from app.evaluation.precision_recall import compute_pr_curve_data

def test_compute_classification_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

def test_compute_classification_metrics_zero_division():
    # All predicted 0, no positive predictions -> precision should be 0.0, not error
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.2])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

def test_confusion_matrix_details():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9]) # TN=1, FP=1, FN=1, TP=1
    cm, details = compute_confusion_matrix_details(y_true, y_prob, threshold=0.5)
    assert details["true_negatives"] == 1
    assert details["false_positives"] == 1
    assert details["false_negatives"] == 1
    assert details["true_positives"] == 1

def test_roc_and_pr_curves():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.7, 0.3, 0.8])
    fpr, tpr, roc_auc = compute_roc_curve_data(y_true, y_prob)
    assert len(fpr) == len(tpr)
    assert 0.0 <= roc_auc <= 1.0
    
    precision, recall, ap = compute_pr_curve_data(y_true, y_prob)
    assert len(precision) == len(recall)
    assert 0.0 <= ap <= 1.0

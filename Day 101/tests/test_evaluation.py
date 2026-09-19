"""Tests for evaluation metrics and confusion matrix."""
import pytest
import numpy as np
from app.evaluation.metrics import calculate_classification_metrics
from app.evaluation.confusion_matrix import get_confusion_matrix_breakdown
from app.evaluation.curves import compute_roc_data, compute_pr_data

def test_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.85])
    
    m = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0

def test_metrics_zero_division():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    m = calculate_classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0

def test_confusion_matrix_breakdown():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    cm = get_confusion_matrix_breakdown(y_true, y_pred)
    assert cm["true_negatives"] == 2
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 1

def test_curves_computation():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    roc = compute_roc_data(y_true, y_proba)
    pr = compute_pr_data(y_true, y_proba)
    assert len(roc["fpr"]) > 0
    assert len(pr["precision"]) > 0

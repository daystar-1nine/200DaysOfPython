"""
Unit tests for evaluation metrics, confusion matrix, and error analysis.
Day 105: Neural NLP & Text Classification.
"""

import numpy as np
import pytest
from app.evaluation.metrics import compute_classification_metrics
from app.evaluation.confusion_matrix import compute_confusion_matrix, format_confusion_matrix
from app.evaluation.error_analysis import ErrorAnalyzer


def test_compute_classification_metrics_perfect():
    y_true = [0, 0, 1, 1]
    y_probs = [0.1, 0.2, 0.9, 0.8]
    metrics = compute_classification_metrics(y_true, y_probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_classification_metrics_zero_division():
    y_true = [0, 0, 0]
    y_probs = [0.1, 0.2, 0.3]  # All predicted 0, no positive class in ground truth
    metrics = compute_classification_metrics(y_true, y_probs)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_confusion_matrix_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["true_negatives"] == 1
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 1


def test_confusion_matrix_formatting():
    cm = {
        "true_negatives": 10,
        "false_positives": 2,
        "false_negatives": 1,
        "true_positives": 5
    }
    table_str = format_confusion_matrix(cm)
    assert "Predicted HAM" in table_str
    assert "Predicted SPAM" in table_str
    assert "10" in table_str


def test_error_analyzer_detection():
    texts = [
        "Normal message",
        "False alarm free win prize",
        "Secret code hidden"
    ]
    y_true = [0, 0, 1]
    y_probs = [0.1, 0.9, 0.2]  # item 1 is FP, item 2 is FN

    df = ErrorAnalyzer.analyze(texts, y_true, y_probs)
    assert len(df) == 2
    assert "False Positive" in df["error_type"].values
    assert "False Negative" in df["error_type"].values

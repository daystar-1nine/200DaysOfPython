"""
Tests for classification metrics and confusion matrix in Day 106.
"""

import numpy as np
import pytest
from app.evaluation.metrics import ClassificationMetrics
from app.evaluation.confusion_matrix import ConfusionMatrixCalculator
from app.evaluation.roc import ROCEvaluator
from app.evaluation.precision_recall import PrecisionRecallEvaluator


def test_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_probs = [0.05, 0.10, 0.90, 0.95]
    metrics = ClassificationMetrics.compute(y_true, y_probs, threshold=0.50)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_metrics_zero_division_resilience():
    y_true = [0, 0, 0, 0]
    y_probs = [0.1, 0.2, 0.1, 0.3]
    # No positives predicted or actual
    metrics = ClassificationMetrics.compute(y_true, y_probs, threshold=0.50)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_confusion_matrix_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cm = ConfusionMatrixCalculator.compute(y_true, y_pred)
    assert cm["tn"] == 1
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 1
    assert cm["false_positive_rate"] == 0.5
    assert cm["false_negative_rate"] == 0.5


def test_roc_evaluator():
    y_true = [0, 0, 1, 1]
    y_probs = [0.1, 0.2, 0.8, 0.9]
    res = ROCEvaluator.compute_curve(y_true, y_probs)
    assert "fpr" in res
    assert "tpr" in res
    assert res["auc"] == 1.0


def test_precision_recall_evaluator():
    y_true = [0, 0, 1, 1]
    y_probs = [0.1, 0.2, 0.8, 0.9]
    res = PrecisionRecallEvaluator.compute_curve(y_true, y_probs)
    assert "precision" in res
    assert "recall" in res
    assert res["average_precision"] == 1.0

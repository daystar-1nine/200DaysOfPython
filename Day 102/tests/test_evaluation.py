"""Tests for evaluation metrics, cross-validation, and confusion matrix."""
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.evaluation.metrics import evaluate_predictions, measure_inference_time
from app.evaluation.cross_validation import run_stratified_cv
from app.evaluation.confusion_matrix import compute_confusion_breakdown
from app.evaluation.curves import extract_roc_curve, extract_pr_curve

def test_evaluate_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.85])
    m = evaluate_predictions(y_true, y_pred, y_score)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0

def test_evaluate_zero_division():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    m = evaluate_predictions(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0

def test_confusion_breakdown():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    cm = compute_confusion_breakdown(y_true, y_pred)
    assert cm["true_negatives"] == 2
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 1

def test_curves_extraction():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    roc = extract_roc_curve(y_true, y_score)
    pr = extract_pr_curve(y_true, y_score)
    assert len(roc["fpr"]) > 0
    assert len(pr["precision"]) > 0

def test_stratified_cv():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    X = np.array(["win prize", "claim cash", "hello friend", "see you"] * 5)
    y = np.array([1, 1, 0, 0] * 5)
    mean_f1, std_f1 = run_stratified_cv(pipe, X, y, n_splits=3)
    assert 0.0 <= mean_f1 <= 1.0
    assert std_f1 >= 0.0

def test_inference_timing():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    X = np.array(["win prize", "claim cash", "hello friend", "see you"])
    y = np.array([1, 1, 0, 0])
    pipe.fit(X, y)
    ms = measure_inference_time(pipe, X)
    assert ms > 0.0

def test_empty_predictions_raises():
    with pytest.raises(ValueError):
        evaluate_predictions(np.array([]), np.array([]))

def test_metrics_avg_precision():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    m = evaluate_predictions(y_true, y_pred, y_score)
    assert "avg_precision" in m
    assert m["avg_precision"] > 0.9

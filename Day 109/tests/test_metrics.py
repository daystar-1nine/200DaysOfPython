import pytest
import numpy as np
from app.evaluation.metrics import compute_classification_metrics
from app.evaluation.confusion import compute_confusion_matrix_details
from app.evaluation.threshold import analyze_thresholds
from app.analysis.entropy import compute_attention_entropy, summarize_attention_distribution

def test_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.95])
    m = compute_classification_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0

def test_metrics_zero_positive_preds():
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.2])
    m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0

def test_confusion_matrix_values():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9]) # TN=1, FP=1, FN=1, TP=1
    cm, details = compute_confusion_matrix_details(y_true, y_prob)
    assert details["true_negatives"] == 1
    assert details["false_positives"] == 1
    assert details["false_negatives"] == 1
    assert details["true_positives"] == 1

def test_threshold_monotonicity():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.7, 0.85, 0.4, 0.9])
    df = analyze_thresholds(y_true, y_prob, model_name="Test")
    assert len(df) == 9
    fps = df["false_positives"].tolist()
    for i in range(len(fps) - 1):
        assert fps[i] >= fps[i+1]

def test_attention_entropy_uniform_vs_peaked():
    # Uniform attention distribution over 4 tokens
    uniform_wts = np.array([0.25, 0.25, 0.25, 0.25])
    # Peaked attention (almost all on token 0)
    peaked_wts = np.array([0.97, 0.01, 0.01, 0.01])
    
    ent_uniform = compute_attention_entropy(uniform_wts)
    ent_peaked = compute_attention_entropy(peaked_wts)
    
    assert ent_uniform > ent_peaked
    assert np.isclose(ent_uniform, np.log(4), atol=1e-3)

def test_summarize_attention_distribution():
    w = np.array([[0.5, 0.5], [0.8, 0.2]])
    summary = summarize_attention_distribution(w)
    assert summary["max_attention"] == 0.8
    assert summary["min_attention"] == 0.2
    assert np.isclose(summary["mean_attention"], 0.5)
    assert "entropy" in summary

import numpy as np
from app.threshold import evaluate_thresholds, find_optimal_threshold_f1, find_optimal_threshold_recall_constrained

def test_evaluate_thresholds():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])
    df_t = evaluate_thresholds(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])
    assert len(df_t) == 3
    assert "F1_Score" in df_t.columns

def test_find_optimal_f1():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    df_t = evaluate_thresholds(y_true, y_prob, thresholds=[0.2, 0.5, 0.8])
    best_t = find_optimal_threshold_f1(df_t)
    assert best_t == 0.5

def test_find_optimal_recall_constrained():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    df_t = evaluate_thresholds(y_true, y_prob, thresholds=[0.2, 0.5, 0.8])
    t = find_optimal_threshold_recall_constrained(df_t, min_recall=0.80)
    assert t in [0.2, 0.5]

"""
Tests for ThresholdAnalyzer in Day 106: RNNs & Sequential Text Learning.
"""

import pandas as pd
from app.evaluation.threshold import ThresholdAnalyzer


def test_threshold_analyzer_dataframe_structure():
    y_true = [0, 0, 1, 1]
    y_probs = [0.1, 0.4, 0.6, 0.9]
    df = ThresholdAnalyzer.evaluate_thresholds(y_true, y_probs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9  # 0.1 to 0.9
    for col in ["threshold", "precision", "recall", "f1", "false_positives", "false_negatives"]:
        assert col in df.columns


def test_threshold_analyzer_custom_thresholds():
    y_true = [0, 1]
    y_probs = [0.2, 0.8]
    custom_th = [0.3, 0.7]
    df = ThresholdAnalyzer.evaluate_thresholds(y_true, y_probs, thresholds=custom_th)
    assert len(df) == 2
    assert df["threshold"].tolist() == [0.3, 0.7]


def test_threshold_monotonic_recall():
    # As threshold increases, recall generally decreases or stays same
    y_true = [0, 0, 1, 1, 1]
    y_probs = [0.1, 0.3, 0.4, 0.7, 0.85]
    df = ThresholdAnalyzer.evaluate_thresholds(y_true, y_probs)
    recalls = df["recall"].tolist()
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1]


def test_threshold_fp_decreases_with_threshold():
    # As threshold increases, false positives decrease or stay same
    y_true = [0, 0, 0, 1]
    y_probs = [0.2, 0.4, 0.6, 0.9]
    df = ThresholdAnalyzer.evaluate_thresholds(y_true, y_probs)
    fps = df["false_positives"].tolist()
    for i in range(len(fps) - 1):
        assert fps[i] >= fps[i + 1]


def test_threshold_extreme_bounds():
    y_true = [0, 1]
    y_probs = [0.0, 1.0]
    df = ThresholdAnalyzer.evaluate_thresholds(y_true, y_probs, thresholds=[0.5])
    assert df["precision"].iloc[0] == 1.0
    assert df["recall"].iloc[0] == 1.0
    assert df["f1"].iloc[0] == 1.0

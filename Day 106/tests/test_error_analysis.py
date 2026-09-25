"""
Tests for ErrorAnalyzer in Day 106: RNNs & Sequential Text Learning.
"""

import pandas as pd
from app.analysis.error_analysis import ErrorAnalyzer


def test_error_analyzer_perfect_predictions_zero_errors():
    messages = ["Hello friend", "Claim your free prize now"]
    y_true = [0, 1]
    y_probs = [0.1, 0.9]
    df = ErrorAnalyzer.analyze(messages, y_true, y_probs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_error_analyzer_detects_false_positive():
    messages = ["Normal message looking like spam: call me now"]
    y_true = [0]
    y_probs = [0.85]  # Predicted spam -> False Positive
    df = ErrorAnalyzer.analyze(messages, y_true, y_probs)
    assert len(df) == 1
    assert df["error_type"].iloc[0] == "False Positive"
    assert df["actual_label"].iloc[0] == "ham"
    assert df["predicted_label"].iloc[0] == "spam"


def test_error_analyzer_detects_false_negative():
    messages = ["Secret lottery winning click here"]
    y_true = [1]
    y_probs = [0.25]  # Predicted ham -> False Negative
    df = ErrorAnalyzer.analyze(messages, y_true, y_probs)
    assert len(df) == 1
    assert df["error_type"].iloc[0] == "False Negative"
    assert df["actual_label"].iloc[0] == "spam"
    assert df["predicted_label"].iloc[0] == "ham"


def test_error_analyzer_metadata_extraction():
    messages = [
        "Visit http://win-lottery.xyz or call 0800111222 to claim £500"
    ]
    y_true = [0]
    y_probs = [0.9]
    df = ErrorAnalyzer.analyze(messages, y_true, y_probs)
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["has_url"]) is True
    assert bool(row["has_phone"]) is True
    assert bool(row["has_currency"]) is True
    assert row["char_length"] > 20
    assert row["word_count"] > 5


def test_error_analyzer_columns():
    messages = ["Hello", "Spam!"]
    y_true = [0, 1]
    y_probs = [0.9, 0.1]
    df = ErrorAnalyzer.analyze(messages, y_true, y_probs)
    expected_cols = [
        "message", "actual_label", "predicted_label", "probability",
        "error_type", "char_length", "word_count", "has_url",
        "has_phone", "has_currency"
    ]
    assert list(df.columns) == expected_cols

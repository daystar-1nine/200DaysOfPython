"""Tests for error analysis and diagnostic extraction."""
import pytest
from app.analysis.error_analysis import extract_prediction_errors, diagnose_errors

def test_extract_prediction_errors():
    texts = ["win cash now", "hey are we meeting", "urgent call"]
    y_true = [1, 0, 1]
    y_pred = [1, 1, 0]  # sample 2 is FP, sample 3 is FN
    y_proba = [0.95, 0.8, 0.4]
    
    err_df = extract_prediction_errors(texts, y_true, y_pred, y_proba)
    assert len(err_df) == 2
    assert "False Positive (Ham as Spam)" in err_df["error_type"].values
    assert "False Negative (Spam as Ham)" in err_df["error_type"].values

def test_diagnose_errors():
    texts = ["URGENT call 09061701461 now"]
    y_true = [1]
    y_pred = [0]
    y_proba = [0.45]
    
    err_df = extract_prediction_errors(texts, y_true, y_pred, y_proba)
    diagnostics = diagnose_errors(err_df)
    assert len(diagnostics) == 1
    assert len(diagnostics[0]["reasons"]) > 0

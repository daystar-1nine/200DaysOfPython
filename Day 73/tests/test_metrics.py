"""
Unit tests for regression evaluation metrics module.
"""
import pytest
import numpy as np
from app.metrics import evaluate_predictions

def test_evaluate_predictions_perfect():
    y = np.array([10.0, 20.0, 30.0])
    res = evaluate_predictions(y, y)
    assert res["mae"] == 0.0
    assert res["mse"] == 0.0
    assert res["rmse"] == 0.0
    assert res["r2"] == 1.0

def test_evaluate_predictions_known_errors():
    y_true = np.array([100.0, 120.0, 140.0])
    y_pred = np.array([90.0, 130.0, 135.0]) # errors: 10, -10, 5
    res = evaluate_predictions(y_true, y_pred)
    assert np.isclose(res["mae"], (10 + 10 + 5) / 3)
    assert np.isclose(res["mse"], (100 + 100 + 25) / 3)
    assert np.isclose(res["rmse"], np.sqrt(res["mse"]))

def test_evaluate_predictions_rmse_ge_mae():
    np.random.seed(42)
    y_true = np.random.uniform(50, 150, 50)
    y_pred = y_true + np.random.normal(0, 10, 50)
    res = evaluate_predictions(y_true, y_pred)
    assert res["rmse"] >= res["mae"]

def test_evaluate_predictions_negative_r2():
    # Terrible predictions worse than mean
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([500.0, 600.0, 700.0])
    res = evaluate_predictions(y_true, y_pred)
    assert res["r2"] < 0.0

def test_evaluate_predictions_scalar_lists():
    res = evaluate_predictions([10, 20], [12, 18])
    assert isinstance(res["mae"], float)
    assert np.isclose(res["mae"], 2.0)

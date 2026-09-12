"""
Unit tests for residual diagnostics engine.
"""
import pytest
import numpy as np
from app.residuals import analyze_residuals

def test_residuals_mean_and_sum_near_zero(clean_linear_arrays):
    from sklearn.linear_model import LinearRegression
    X, y = clean_linear_arrays
    model = LinearRegression().fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    
    res = analyze_residuals(y, y_pred)
    assert np.isclose(res["mean_residual"], 0.0, atol=1e-10)
    assert np.isclose(res["sum_residual"], 0.0, atol=1e-10)

def test_residuals_array_length():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([11, 19, 31, 39])
    res = analyze_residuals(y_true, y_pred)
    assert len(res["residuals"]) == 4

def test_residuals_homoscedasticity_detection():
    # Homoscedastic synthetic data
    np.random.seed(42)
    n = 100
    y_pred = np.linspace(50, 200, n)
    residuals = np.random.normal(0, 5, n)
    y_true = y_pred + residuals
    res = analyze_residuals(y_true, y_pred)
    assert res["is_homoscedastic"] is True

def test_residuals_heteroscedasticity_detection():
    # Heteroscedastic data: error standard deviation grows with y_pred
    np.random.seed(42)
    n = 200
    y_pred = np.linspace(10, 200, n)
    # Variance expands from 1 to 50
    noise = np.random.normal(0, 1, n) * (y_pred * 0.4)
    y_true = y_pred + noise
    res = analyze_residuals(y_true, y_pred)
    assert res["variance_ratio"] > 3.0
    assert res["is_homoscedastic"] is False

def test_residuals_skewness_calculation():
    y_true = np.array([10, 20, 30, 100]) # 1 huge positive outlier
    y_pred = np.array([10, 20, 30, 40])
    res = analyze_residuals(y_true, y_pred)
    assert res["skewness"] > 0

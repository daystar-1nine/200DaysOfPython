import pytest
import numpy as np

def calculate_residuals(y_true, y_pred):
    return y_true - y_pred

def test_calculate_residuals_formula():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([9, 21, 30])
    res = calculate_residuals(y_true, y_pred)
    assert np.allclose(res, np.array([1, -1, 0]))

def test_residual_stats_keys():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([9, 21, 30])
    res = calculate_residuals(y_true, y_pred)
    stats = {'mean': np.mean(res), 'std': np.std(res)}
    assert 'mean' in stats
    assert 'std' in stats

def test_residual_stats_mean_near_zero_ols(dummy_xy):
    from sklearn.linear_model import LinearRegression
    X, y = dummy_xy
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    res = calculate_residuals(y, y_pred)
    assert np.isclose(np.mean(res), 0, atol=1e-7)

def test_residuals_length_matches():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([9, 21, 30])
    res = calculate_residuals(y_true, y_pred)
    assert len(res) == len(y_true)

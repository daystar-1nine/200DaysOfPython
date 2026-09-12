import pytest
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, p):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Adjusted_R2': adj_r2}

def test_mae_non_negative():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])
    res = evaluate_model(y_true, y_pred, 1)
    assert res['MAE'] >= 0

def test_mse_non_negative():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])
    res = evaluate_model(y_true, y_pred, 1)
    assert res['MSE'] >= 0

def test_rmse_matches_sqrt_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])
    res = evaluate_model(y_true, y_pred, 1)
    assert np.isclose(res['RMSE'], np.sqrt(res['MSE']))

def test_r2_bounded():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])
    res = evaluate_model(y_true, y_pred, 1)
    assert res['R2'] <= 1.0

def test_adjusted_r2_penalty():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 4.1, 4.9])
    res1 = evaluate_model(y_true, y_pred, 1)
    res2 = evaluate_model(y_true, y_pred, 3)
    assert res2['Adjusted_R2'] < res1['Adjusted_R2']

def test_evaluate_model_returns_dict():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])
    res = evaluate_model(y_true, y_pred, 1)
    assert isinstance(res, dict)
    assert all(k in res for k in ['MAE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2'])

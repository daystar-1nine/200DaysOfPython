import pytest
import numpy as np

try:
    from app.metrics import calculate_mae, calculate_mse, calculate_rmse, calculate_r2, calculate_adjusted_r2, format_metrics_report
except ImportError:
    from metrics import calculate_mae, calculate_mse, calculate_rmse, calculate_r2, calculate_adjusted_r2, format_metrics_report

@pytest.fixture
def eval_data():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    return y_true, y_pred

def test_mae_non_negative(eval_data):
    y_true, y_pred = eval_data
    assert calculate_mae(y_true, y_pred) >= 0

def test_mse_non_negative(eval_data):
    y_true, y_pred = eval_data
    assert calculate_mse(y_true, y_pred) >= 0

def test_rmse_equals_sqrt_mse(eval_data):
    y_true, y_pred = eval_data
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(mse))

def test_r2_between_zero_and_one_for_good_model(eval_data):
    y_true, y_pred = eval_data
    r2 = calculate_r2(y_true, y_pred)
    assert r2 <= 1.0

def test_adjusted_r2_less_than_r2_when_features_added(eval_data):
    y_true, y_pred = eval_data
    r2 = calculate_r2(y_true, y_pred)
    n = len(y_true)
    p = 2
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    assert adj_r2 <= r2

def test_adjusted_r2_formula_manual():
    r2 = 0.9
    n = 100
    p = 5
    expected = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    assert np.isclose(adj_r2, expected)

def test_format_metrics_report_is_string(eval_data):
    y_true, y_pred = eval_data
    report = format_metrics_report(y_true, y_pred, n=100, p=5)
    assert isinstance(report, str)
    assert len(report) > 0

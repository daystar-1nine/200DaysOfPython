import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

try:
    from app.residuals import calculate_residuals, residual_stats
except ImportError:
    from residuals import calculate_residuals, residual_stats

@pytest.fixture
def ols_data():
    X = np.random.rand(100, 2)
    y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100) * 0.1
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    return y, preds

def test_residuals_mean_near_zero_for_ols(ols_data):
    y_true, y_pred = ols_data
    residuals = calculate_residuals(y_true, y_pred)
    assert np.isclose(np.mean(residuals), 0, atol=1e-7)

def test_residuals_length_matches_predictions(ols_data):
    y_true, y_pred = ols_data
    residuals = calculate_residuals(y_true, y_pred)
    assert len(residuals) == len(y_pred)

def test_residuals_formula_correct(ols_data):
    y_true, y_pred = ols_data
    residuals = calculate_residuals(y_true, y_pred)
    np.testing.assert_array_almost_equal(residuals, y_true - y_pred)

def test_residual_stats_returns_dict(ols_data):
    y_true, y_pred = ols_data
    residuals = calculate_residuals(y_true, y_pred)
    stats = residual_stats(residuals)
    assert isinstance(stats, dict)

def test_residual_stats_has_required_keys(ols_data):
    y_true, y_pred = ols_data
    residuals = calculate_residuals(y_true, y_pred)
    stats = residual_stats(residuals)
    assert 'mean' in stats
    assert 'std' in stats
    assert 'skewness' in stats

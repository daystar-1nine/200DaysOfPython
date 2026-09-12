"""
Unit tests for SimpleLinearRegressor module.
"""
import pytest
import numpy as np
from app.regression import SimpleLinearRegressor

def test_regressor_fit_and_parameters(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor(fit_intercept=True)
    reg.fit(X, y)
    # y = 10 + 2.0 * X
    assert reg.is_fitted is True
    assert np.isclose(reg.slope, 2.0)
    assert np.isclose(reg.intercept, 10.0)

def test_regressor_predict_shape(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor()
    reg.fit(X, y)
    preds = reg.predict([60, 70, 80])
    assert len(preds) == 3
    assert np.isclose(preds[0], 130.0)

def test_regressor_predict_before_fit_raises():
    reg = SimpleLinearRegressor()
    with pytest.raises(RuntimeError, match="not fitted"):
        reg.predict([10, 20])

def test_regressor_equation_format(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor().fit(X, y)
    eq = reg.get_equation("Spend", "Revenue")
    assert "Revenue =" in eq
    assert "* Spend" in eq

def test_regressor_get_params_dict(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor().fit(X, y)
    params = reg.get_params()
    assert "slope" in params and "intercept" in params

def test_regressor_no_intercept():
    X = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    reg = SimpleLinearRegressor(fit_intercept=False).fit(X, y)
    assert reg.intercept == 0.0
    assert np.isclose(reg.slope, 2.0)

import pytest
from sklearn.linear_model import Ridge

def test_ridge_fits(dummy_xy):
    X, y = dummy_xy
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    assert hasattr(model, 'coef_')

def test_ridge_coefficients_shrink_with_alpha(dummy_xy):
    X, y = dummy_xy
    model_low = Ridge(alpha=0.1).fit(X, y)
    model_high = Ridge(alpha=100.0).fit(X, y)
    import numpy as np
    assert np.sum(np.abs(model_high.coef_)) < np.sum(np.abs(model_low.coef_))

def test_ridge_predict_shape(dummy_xy):
    X, y = dummy_xy
    model = Ridge(alpha=1.0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

def test_ridge_score_valid(dummy_xy):
    X, y = dummy_xy
    model = Ridge(alpha=1.0).fit(X, y)
    score = model.score(X, y)
    assert score <= 1.0

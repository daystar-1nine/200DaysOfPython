import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    return {
        'scores': rmse_scores,
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores)
    }

def test_cv_returns_dict(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    res = cross_validate_model(model, X, y)
    assert isinstance(res, dict)

def test_cv_scores_length(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    cv_folds = 5
    res = cross_validate_model(model, X, y, cv=cv_folds)
    assert len(res['scores']) == cv_folds

def test_cv_mean_rmse_positive(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    res = cross_validate_model(model, X, y)
    assert res['mean_rmse'] >= 0

def test_cv_std_rmse_non_negative(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    res = cross_validate_model(model, X, y)
    assert res['std_rmse'] >= 0

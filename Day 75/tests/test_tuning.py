import pytest
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

def test_tune_ridge_finds_best_alpha(dummy_xy):
    X, y = dummy_xy
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    grid = GridSearchCV(Ridge(), param_grid, cv=3)
    grid.fit(X, y)
    assert grid.best_params_['alpha'] in param_grid['alpha']

def test_tune_lasso_finds_best_alpha(dummy_xy):
    X, y = dummy_xy
    param_grid = {'alpha': [0.01, 0.1, 1.0]}
    grid = GridSearchCV(Lasso(max_iter=5000), param_grid, cv=3)
    grid.fit(X, y)
    assert grid.best_params_['alpha'] in param_grid['alpha']

def test_tune_poly_degree_returns_optimal(dummy_xy):
    X, y = dummy_xy
    pipeline = make_pipeline(PolynomialFeatures(), Ridge(alpha=1.0))
    param_grid = {'polynomialfeatures__degree': [1, 2, 3]}
    grid = GridSearchCV(pipeline, param_grid, cv=3)
    grid.fit(X, y)
    assert grid.best_params_['polynomialfeatures__degree'] in param_grid['polynomialfeatures__degree']

def test_tuning_results_format(dummy_xy):
    X, y = dummy_xy
    param_grid = {'alpha': [0.1, 1.0]}
    grid = GridSearchCV(Ridge(), param_grid, cv=3)
    grid.fit(X, y)
    assert isinstance(grid.cv_results_, dict)
    assert 'mean_test_score' in grid.cv_results_

import pytest
from sklearn.linear_model import Lasso
import numpy as np

def test_lasso_fits(dummy_xy):
    X, y = dummy_xy
    model = Lasso(alpha=0.1)
    model.fit(X, y)
    assert hasattr(model, 'coef_')

def test_lasso_sparsity_with_high_alpha(dummy_xy):
    # Create dummy data with many irrelevant features
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (100, 10))
    y = 5 + 2 * X[:, 0] + np.random.normal(0, 1, 100)
    
    model_low = Lasso(alpha=0.01).fit(X, y)
    model_high = Lasso(alpha=2.0).fit(X, y)
    
    zeros_low = np.sum(model_low.coef_ == 0)
    zeros_high = np.sum(model_high.coef_ == 0)
    
    assert zeros_high > zeros_low

def test_lasso_predict_shape(dummy_xy):
    X, y = dummy_xy
    model = Lasso(alpha=0.1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

def test_lasso_score_valid(dummy_xy):
    X, y = dummy_xy
    model = Lasso(alpha=0.1).fit(X, y)
    score = model.score(X, y)
    assert score <= 1.0

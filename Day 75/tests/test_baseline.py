import pytest
import numpy as np
from sklearn.dummy import DummyRegressor

def test_baseline_fits(dummy_xy):
    X, y = dummy_xy
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    assert hasattr(model, 'constant_')

def test_baseline_predicts_mean(dummy_xy):
    X, y = dummy_xy
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, np.mean(y))

def test_baseline_predict_shape(dummy_xy):
    X, y = dummy_xy
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

def test_baseline_score_leq_zero_or_valid(dummy_xy):
    X, y = dummy_xy
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    score = model.score(X, y)
    assert score <= 1.0

import pytest
from sklearn.linear_model import LinearRegression

def test_linear_fits(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    model.fit(X, y)
    assert hasattr(model, 'coef_')

def test_linear_predict_shape(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

def test_linear_score_valid(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    assert score <= 1.0

def test_linear_has_coefficients(dummy_xy):
    X, y = dummy_xy
    model = LinearRegression()
    model.fit(X, y)
    assert len(model.coef_) == X.shape[1]

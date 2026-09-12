import pytest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def test_polynomial_fits_deg2(dummy_xy):
    X, y = dummy_xy
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    assert hasattr(model.named_steps['linearregression'], 'coef_')

def test_polynomial_fits_deg3(dummy_xy):
    X, y = dummy_xy
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(X, y)
    assert hasattr(model.named_steps['linearregression'], 'coef_')

def test_polynomial_captures_quadratic_better_than_linear(dummy_xy):
    X, y = dummy_xy
    linear = LinearRegression().fit(X, y)
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y)
    assert poly.score(X, y) > linear.score(X, y)

def test_polynomial_predict_shape(dummy_xy):
    X, y = dummy_xy
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

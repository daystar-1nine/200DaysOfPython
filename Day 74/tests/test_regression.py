import pytest
import pandas as pd
import numpy as np

try:
    from app.regression import train_model, evaluate_model, get_coefficients
except ImportError:
    from regression import train_model, evaluate_model, get_coefficients

@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(100) * 0.1
    return X, y

def test_regressor_fits_without_error(dummy_data):
    X, y = dummy_data
    model = train_model(X, y)
    assert model is not None

def test_regressor_predict_returns_array(dummy_data):
    X, y = dummy_data
    model = train_model(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)

def test_regressor_predict_length_matches_test(dummy_data):
    X, y = dummy_data
    model = train_model(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)

def test_regressor_score_between_zero_and_one(dummy_data):
    X, y = dummy_data
    model = train_model(X, y)
    score = model.score(X, y)
    assert 0.0 <= score <= 1.0

def test_regressor_coefficients_dataframe(dummy_data):
    X, y = dummy_data
    model = train_model(X, y)
    feature_names = ['f1', 'f2', 'f3']
    coef_df = get_coefficients(model, feature_names)
    assert isinstance(coef_df, pd.DataFrame)
    assert 'Feature' in coef_df.columns
    assert 'Coefficient' in coef_df.columns

def test_regressor_intercept_is_float(dummy_data):
    X, y = dummy_data
    model = train_model(X, y)
    assert isinstance(model.intercept_, float)

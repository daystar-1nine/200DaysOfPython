import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_model_fits_successfully(dummy_classification_data):
    X, y, _ = dummy_classification_data
    model = LogisticRegression()
    model.fit(X, y)
    assert hasattr(model, 'coef_')

def test_model_predict_returns_binary(dummy_classification_data):
    X, y, _ = dummy_classification_data
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

def test_model_predict_proba_range_zero_one(dummy_classification_data):
    X, y, _ = dummy_classification_data
    model = LogisticRegression().fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()

def test_model_coefficients_dataframe(dummy_classification_data):
    X, y, _ = dummy_classification_data
    model = LogisticRegression().fit(X, y)
    coef_df = pd.DataFrame({'feature': X.columns, 'coef': model.coef_[0]})
    assert len(coef_df) == X.shape[1]

def test_model_odds_ratio_positive(dummy_classification_data):
    X, y, _ = dummy_classification_data
    model = LogisticRegression().fit(X, y)
    odds_ratios = np.exp(model.coef_[0])
    assert (odds_ratios > 0).all()

"""Tests for classification models and pipelines."""
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from app.models.baseline import get_baseline_model
from app.models.logistic import get_logistic_model
from app.models.naive_bayes import get_naive_bayes_model

def test_dummy_baseline_pipeline():
    X = ["ham message", "another ham", "spam prize", "ham hello"]
    y = np.array([0, 0, 1, 0])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", get_baseline_model())
    ])
    pipe.fit(X, y)
    preds = pipe.predict(["any text"])
    assert preds[0] == 0

def test_logistic_regression_pipeline():
    X = ["win cash free money", "claim prize urgent", "hello friend lunch", "how are you doing"]
    y = np.array([1, 1, 0, 0])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", get_logistic_model(C=1.0))
    ])
    pipe.fit(X, y)
    preds = pipe.predict(["win prize", "hello friend"])
    assert preds[0] == 1
    assert preds[1] == 0

def test_naive_bayes_pipeline():
    X = ["win cash free", "prize money claim", "lunch tomorrow meeting", "notes for exam"]
    y = np.array([1, 1, 0, 0])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", get_naive_bayes_model(alpha=1.0))
    ])
    pipe.fit(X, y)
    preds = pipe.predict(["free cash prize", "exam notes"])
    assert preds[0] == 1
    assert preds[1] == 0

def test_model_predict_proba():
    X = ["win prize", "claim cash", "hello lunch", "see you"]
    y = np.array([1, 1, 0, 0])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", get_logistic_model())
    ])
    pipe.fit(X, y)
    probas = pipe.predict_proba(["win cash"])
    assert probas.shape == (1, 2)
    assert np.isclose(np.sum(probas), 1.0)

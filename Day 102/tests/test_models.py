"""Tests for model factories and pipeline execution."""
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from app.models.naive_bayes import get_naive_bayes_model
from app.models.logistic_regression import get_logistic_model
from app.models.linear_svm import get_linear_svm_model

def test_naive_bayes_training():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", get_naive_bayes_model())])
    X = ["win cash prize", "hey lunch tomorrow"]
    y = np.array([1, 0])
    pipe.fit(X, y)
    assert pipe.predict(["prize"])[0] == 1

def test_logistic_regression_training():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", get_logistic_model(C=1.0))])
    X = ["claim free reward", "let us meet"]
    y = np.array([1, 0])
    pipe.fit(X, y)
    assert pipe.predict(["free reward"])[0] == 1

def test_linear_svm_calibrated():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", get_linear_svm_model(C=1.0, calibrate=True))])
    X = ["win prize", "claim reward", "free cash", "hello friend", "see you", "lunch today"] * 4
    y = np.array([1, 1, 1, 0, 0, 0] * 4)
    pipe.fit(X, y)
    preds = pipe.predict(["win cash"])
    probas = pipe.predict_proba(["win cash"])
    assert preds[0] == 1
    assert probas.shape == (1, 2)

def test_linear_svm_uncalibrated():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", get_linear_svm_model(C=1.0, calibrate=False))])
    X = ["win prize", "claim reward", "hello friend", "see you"]
    y = np.array([1, 1, 0, 0])
    pipe.fit(X, y)
    scores = pipe.decision_function(["win prize"])
    assert scores.shape == (1,)

def test_logistic_predict_proba():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", get_logistic_model())])
    X = ["win prize", "hello friend"]
    y = np.array([1, 0])
    pipe.fit(X, y)
    proba = pipe.predict_proba(["win prize"])
    assert np.isclose(np.sum(proba), 1.0)

def test_naive_bayes_alpha():
    clf = get_naive_bayes_model(alpha=0.5)
    assert clf.alpha == 0.5

def test_logistic_regularization_c():
    clf = get_logistic_model(C=10.0)
    assert clf.C == 10.0

def test_linear_svm_c():
    clf = get_linear_svm_model(C=5.0, calibrate=False)
    assert clf.C == 5.0

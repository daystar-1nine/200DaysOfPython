"""Tests for feature analysis, error extraction, and data audit."""
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.analysis.feature_analysis import extract_top_features
from app.analysis.error_analysis import extract_classification_errors
from app.data.audit import audit_dataset
from app.data.validator import validate_data

def test_extract_top_features():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())])
    X = ["win cash prize free", "claim reward", "hello friend lunch", "how are you"]
    y = np.array([1, 1, 0, 0])
    pipe.fit(X, y)
    df_spam, df_ham = extract_top_features(pipe, top_n=2)
    assert len(df_spam) == 2
    assert len(df_ham) == 2
    assert df_spam.iloc[0]["association"] == "spam"
    assert df_ham.iloc[0]["association"] == "ham"

def test_extract_classification_errors():
    texts = ["win cash", "hello friend", "free prize"]
    y_true = [1, 0, 1]
    y_pred = [1, 1, 0]
    y_score = [0.9, 0.8, 0.4]
    df_err = extract_classification_errors(texts, y_true, y_pred, y_score, "TestModel")
    assert len(df_err) == 2
    assert "False Positive" in df_err["error_type"].values
    assert "False Negative" in df_err["error_type"].values

def test_audit_dataset():
    df = pd.DataFrame({
        "label": ["ham", "spam", "ham"],
        "text": ["hello", "win cash prize", "how are you"]
    })
    res = audit_dataset(df)
    assert res["total_records"] == 3
    assert res["num_classes"] == 2
    assert res["char_length"]["min"] == 5

def test_validator_valid():
    df = pd.DataFrame({"label": ["ham", "spam"], "text": ["hello", "win"]})
    assert validate_data(df)["is_valid"] is True

def test_validator_invalid_class():
    df = pd.DataFrame({"label": ["unknown"], "text": ["hello"]})
    assert validate_data(df)["is_valid"] is False

def test_validator_nulls():
    df = pd.DataFrame({"label": ["ham", None], "text": [None, "hello"]})
    assert validate_data(df)["is_valid"] is False

def test_validator_missing_columns():
    df = pd.DataFrame({"col1": [1, 2]})
    assert validate_data(df)["is_valid"] is False

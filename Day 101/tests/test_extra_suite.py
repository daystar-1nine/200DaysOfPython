"""Additional comprehensive tests for Day 101 modules."""
import pytest
import pandas as pd
import numpy as np
from app.data.validator import validate_dataset
from app.preprocessing.normalization import normalize_text
from app.features.bow import BagOfWords
from app.features.tfidf import TFIDFScratch
from app.evaluation.metrics import calculate_classification_metrics

def test_validator_detects_imbalance():
    df = pd.DataFrame({
        "label": ["ham"] * 90 + ["spam"] * 10,
        "text": ["hello"] * 100
    })
    res = validate_dataset(df)
    assert res["is_valid"] is True
    assert res["class_imbalance_detected"] is True

def test_validator_empty_raises():
    with pytest.raises(ValueError):
        validate_dataset(pd.DataFrame())

def test_normalize_tabs_and_newlines():
    text = "Line 1\n\tLine 2   \r\nLine 3"
    assert normalize_text(text) == "line 1 line 2 line 3"

def test_bow_min_freq_filtering():
    docs = ["apple banana", "apple orange", "apple kiwi"]
    # 'apple' appears 3 times, others appear 1 time
    bow = BagOfWords(min_freq=2)
    bow.fit(docs)
    assert "apple" in bow.vocabulary_
    assert "banana" not in bow.vocabulary_
    assert "orange" not in bow.vocabulary_

def test_bow_get_feature_names_order():
    docs = ["zebra apple cat"]
    bow = BagOfWords().fit(docs)
    feature_names = bow.get_feature_names()
    assert feature_names == ["apple", "cat", "zebra"]

def test_tfidf_scratch_manual_idf():
    # 2 docs: doc1 has "python", doc2 has "python" and "code"
    docs = ["python", "python code"]
    tfidf = TFIDFScratch().fit(docs)
    # "python" appears in 2 docs, "code" appears in 1 doc
    # idf of code should be strictly greater than idf of python
    idx_python = tfidf.vocabulary_["python"]
    idx_code = tfidf.vocabulary_["code"]
    assert tfidf.idf_[idx_code] > tfidf.idf_[idx_python]

def test_classification_metrics_empty_inputs():
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError):
        calculate_classification_metrics(y_true, y_pred)

def test_classification_metrics_average_precision():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    res = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert "avg_precision" in res
    assert res["avg_precision"] > 0.9

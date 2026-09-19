"""Tests for TF-IDF feature extraction."""
import pytest
import numpy as np
from app.features.tfidf import TFIDFScratch, get_sklearn_tfidf

def test_tfidf_scratch_fit_transform():
    docs = ["information retrieval", "natural language processing", "information processing"]
    tfidf = TFIDFScratch()
    matrix = tfidf.fit_transform(docs)
    
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(tfidf.vocabulary_)
    # Row norms should be ~1.0 due to L2 normalization
    norms = np.linalg.norm(matrix, axis=1)
    for norm in norms:
        assert pytest.approx(norm, abs=1e-5) == 1.0

def test_tfidf_scratch_unfitted_raises():
    tfidf = TFIDFScratch()
    with pytest.raises(ValueError):
        tfidf.transform(["sample"])

def test_sklearn_tfidf_wrapper():
    vec = get_sklearn_tfidf(ngram_range=(1, 2), min_df=1)
    docs = ["win cash prize now", "urgent call now"]
    X = vec.fit_transform(docs)
    assert X.shape[0] == 2
    assert X.shape[1] > 0

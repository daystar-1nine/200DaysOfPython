"""Tests for from-scratch TF-IDF implementation."""
import pytest
import numpy as np
from app.features.scratch_tfidf import TFIDFScratch

def test_scratch_fit_transform():
    docs = ["python data science", "python machine learning"]
    tfidf = TFIDFScratch()
    matrix = tfidf.fit_transform(docs)
    assert matrix.shape == (2, len(tfidf.vocabulary_))
    assert "python" in tfidf.vocabulary_

def test_scratch_l2_normalization():
    docs = ["hello world", "test sample doc"]
    tfidf = TFIDFScratch()
    matrix = tfidf.fit_transform(docs)
    norms = np.linalg.norm(matrix, axis=1)
    for n in norms:
        assert pytest.approx(n, abs=1e-5) == 1.0

def test_scratch_sublinear_tf():
    docs = ["cat cat cat dog"]
    tfidf = TFIDFScratch(sublinear_tf=True)
    matrix = tfidf.fit_transform(docs)
    assert matrix.shape[0] == 1

def test_scratch_min_df():
    docs = ["apple orange", "apple kiwi", "apple pear"]
    tfidf = TFIDFScratch(min_df=2)
    tfidf.fit(docs)
    assert "apple" in tfidf.vocabulary_
    assert "orange" not in tfidf.vocabulary_

def test_scratch_unseen_words():
    docs = ["cat dog"]
    tfidf = TFIDFScratch().fit(docs)
    test_mat = tfidf.transform(["bird cat"])
    assert test_mat.shape == (1, 2)
    assert test_mat[0, tfidf.vocabulary_["cat"]] > 0

def test_scratch_empty_docs():
    docs = ["hello world"]
    tfidf = TFIDFScratch().fit(docs)
    test_mat = tfidf.transform([""])
    assert np.all(test_mat == 0)

def test_scratch_unfitted_raises():
    tfidf = TFIDFScratch()
    with pytest.raises(ValueError):
        tfidf.transform(["sample"])

def test_scratch_get_feature_names():
    docs = ["zebra apple monkey"]
    tfidf = TFIDFScratch().fit(docs)
    names = tfidf.get_feature_names()
    assert names == ["apple", "monkey", "zebra"]

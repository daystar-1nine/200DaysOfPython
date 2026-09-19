"""Tests for from-scratch Bag of Words."""
import pytest
import numpy as np
from app.features.bow import BagOfWords

def test_bow_fit_transform():
    docs = ["I love Python", "I love Data Science"]
    bow = BagOfWords()
    dtm = bow.fit_transform(docs)
    
    assert isinstance(dtm, np.ndarray)
    assert dtm.shape[0] == 2
    assert "love" in bow.vocabulary_
    assert "python" in bow.vocabulary_
    assert "data" in bow.vocabulary_

def test_bow_counts():
    docs = ["cat dog cat bird"]
    bow = BagOfWords()
    dtm = bow.fit_transform(docs)
    cat_idx = bow.vocabulary_["cat"]
    dog_idx = bow.vocabulary_["dog"]
    assert dtm[0, cat_idx] == 2
    assert dtm[0, dog_idx] == 1

def test_bow_binary_mode():
    docs = ["cat cat cat"]
    bow = BagOfWords(binary=True)
    dtm = bow.fit_transform(docs)
    assert dtm[0, bow.vocabulary_["cat"]] == 1

def test_bow_unseen_words():
    train_docs = ["apple orange"]
    bow = BagOfWords().fit(train_docs)
    test_dtm = bow.transform(["banana apple"])
    assert test_dtm.shape == (1, 2)
    assert test_dtm[0, bow.vocabulary_["apple"]] == 1

def test_bow_unfitted_transform_raises():
    bow = BagOfWords()
    with pytest.raises(ValueError):
        bow.transform(["hello"])

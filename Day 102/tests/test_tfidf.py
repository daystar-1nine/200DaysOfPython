"""Tests for Scikit-Learn TF-IDF vectorizer builders."""
import pytest
from app.features.tfidf import get_word_tfidf, get_char_tfidf

def test_word_tfidf_builder():
    vec = get_word_tfidf(ngram_range=(1, 2), min_df=1)
    docs = ["quick brown fox", "jumped over fox"]
    X = vec.fit_transform(docs)
    assert X.shape[0] == 2
    assert "quick" in vec.vocabulary_

def test_char_tfidf_builder():
    vec = get_char_tfidf(ngram_range=(3, 3), min_df=1)
    docs = ["learning", "training"]
    X = vec.fit_transform(docs)
    assert X.shape[0] == 2
    assert "ing" in vec.vocabulary_

def test_word_tfidf_min_df():
    vec = get_word_tfidf(min_df=2)
    docs = ["apple orange", "apple banana", "apple kiwi"]
    X = vec.fit_transform(docs)
    assert "apple" in vec.vocabulary_
    assert "orange" not in vec.vocabulary_

def test_word_tfidf_max_features():
    vec = get_word_tfidf(min_df=1, max_features=3)
    docs = ["apple banana cherry date elderberry fig"]
    X = vec.fit_transform(docs)
    assert X.shape[1] <= 3

def test_word_tfidf_sublinear_tf():
    vec = get_word_tfidf(sublinear_tf=True, min_df=1)
    docs = ["word word word test"]
    X = vec.fit_transform(docs)
    assert X.shape[0] == 1

def test_char_tfidf_sublinear_tf():
    vec = get_char_tfidf(sublinear_tf=True, min_df=1)
    docs = ["data science"]
    X = vec.fit_transform(docs)
    assert X.shape[0] == 1

def test_word_tfidf_ngram_range():
    vec = get_word_tfidf(ngram_range=(1, 3), min_df=1)
    docs = ["deep neural network"]
    vec.fit(docs)
    assert "deep neural" in vec.vocabulary_
    assert "deep neural network" in vec.vocabulary_

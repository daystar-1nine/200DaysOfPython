"""Tests for n-gram generation."""
import pytest
from app.preprocessing.ngrams import generate_ngrams

def test_generate_unigrams():
    tokens = ["I", "love", "Python"]
    assert generate_ngrams(tokens, 1) == [("I",), ("love",), ("Python",)]

def test_generate_bigrams():
    tokens = ["I", "love", "machine", "learning"]
    expected = [("I", "love"), ("love", "machine"), ("machine", "learning")]
    assert generate_ngrams(tokens, 2) == expected

def test_generate_trigrams():
    tokens = ["deep", "learning", "neural", "networks"]
    expected = [("deep", "learning", "neural"), ("learning", "neural", "networks")]
    assert generate_ngrams(tokens, 3) == expected

def test_generate_ngrams_longer_than_tokens():
    assert generate_ngrams(["a", "b"], 3) == []

def test_generate_ngrams_invalid_n():
    with pytest.raises(ValueError):
        generate_ngrams(["a", "b"], 0)

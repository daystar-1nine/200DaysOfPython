"""Tests for word and character n-gram generation."""
import pytest
from app.preprocessing.tokenizer import word_tokenize, word_ngrams, char_ngrams

def test_word_ngrams_unigram():
    tokens = ["nlp", "is", "fun"]
    assert word_ngrams(tokens, 1) == [("nlp",), ("is",), ("fun",)]

def test_word_ngrams_bigram():
    tokens = ["machine", "learning", "model"]
    assert word_ngrams(tokens, 2) == [("machine", "learning"), ("learning", "model")]

def test_word_ngrams_trigram():
    tokens = ["deep", "neural", "network", "layer"]
    assert word_ngrams(tokens, 3) == [("deep", "neural", "network"), ("neural", "network", "layer")]

def test_word_ngrams_longer_than_tokens():
    assert word_ngrams(["a", "b"], 4) == []

def test_word_ngrams_invalid_n():
    with pytest.raises(ValueError):
        word_ngrams(["a", "b"], 0)

def test_char_ngrams_trigram():
    res = char_ngrams("python", 3)
    assert res == ["pyt", "yth", "tho", "hon"]

def test_char_ngrams_longer_than_text():
    assert char_ngrams("hi", 4) == []

def test_char_ngrams_invalid_n():
    with pytest.raises(ValueError):
        char_ngrams("test", 0)

"""Tests for tokenization utilities."""
import pytest
from app.data.tokenizer import tokenize_text, tokenize_corpus

def test_tokenize_text_normal():
    assert tokenize_text("The quick brown fox") == ["the", "quick", "brown", "fox"]

def test_tokenize_text_empty():
    assert tokenize_text("") == []
    assert tokenize_text(None) == []

def test_tokenize_text_punctuation():
    assert tokenize_text("Hello, world! Word2Vec is cool.") == ["hello", "world", "word2vec", "is", "cool"]

def test_tokenize_text_whitespace():
    assert tokenize_text("  hello   world  \n\t  test  ") == ["hello", "world", "test"]

def test_tokenize_text_non_string():
    assert tokenize_text(12345) == ["12345"]

def test_tokenize_corpus():
    corpus = ["cat drinks milk", "dog drinks water"]
    res = tokenize_corpus(corpus)
    assert len(res) == 2
    assert res[0] == ["cat", "drinks", "milk"]

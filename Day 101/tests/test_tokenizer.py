"""Tests for tokenization utilities."""
import pytest
from app.preprocessing.tokenization import tokenize

def test_tokenize_simple():
    assert tokenize("I love Python") == ["i", "love", "python"]

def test_tokenize_with_punctuation():
    assert tokenize("Hello, world! NLP is great.") == ["hello", "world", "nlp", "is", "great"]

def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []

def test_tokenize_numbers():
    assert tokenize("Call 0800 1234 today") == ["call", "0800", "1234", "today"]

def test_tokenize_mixed_case():
    assert tokenize("PyThOn anD DaTa") == ["python", "and", "data"]

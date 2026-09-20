"""
Unit tests for TextCleaner and Tokenizer.
"""

import pytest
from app.preprocessing.cleaner import TextCleaner
from app.preprocessing.tokenizer import Tokenizer


def test_cleaner_lowercasing():
    cleaner = TextCleaner()
    assert cleaner.clean("PyThOn PrOgRaMmInG") == "python programming"


def test_cleaner_punctuation():
    cleaner = TextCleaner()
    cleaned = cleaner.clean("Hello, world! How's it going?")
    assert "," not in cleaned and "!" not in cleaned and "?" not in cleaned
    assert "hello world how s it going" in cleaned


def test_cleaner_whitespace():
    cleaner = TextCleaner()
    assert cleaner.clean("   multiple    spaces   \n\t") == "multiple spaces"


def test_cleaner_stopwords():
    cleaner = TextCleaner(remove_stopwords=True)
    cleaned = cleaner.clean("this is a test of the search engine")
    tokens = cleaned.split()
    assert "this" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens
    assert "test" in tokens
    assert "search" in tokens
    assert "engine" in tokens


def test_cleaner_non_string():
    cleaner = TextCleaner()
    assert cleaner.clean(None) == ""
    assert cleaner.clean(12345) == ""


def test_tokenizer_basic():
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Python and data science are awesome!")
    assert tokens == ["Python", "and", "data", "science", "are", "awesome"]


def test_tokenizer_min_len():
    tokenizer = Tokenizer(min_token_len=3)
    tokens = tokenizer.tokenize("a in to Python data")
    assert tokens == ["Python", "data"]


def test_tokenizer_empty():
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []
    assert tokenizer.tokenize(None) == []


def test_tokenizer_ngrams():
    tokens = ["deep", "neural", "networks", "learn"]
    bigrams = Tokenizer.get_ngrams(tokens, n=2)
    assert bigrams == ["deep neural", "neural networks", "networks learn"]

    trigrams = Tokenizer.get_ngrams(tokens, n=3)
    assert trigrams == ["deep neural networks", "neural networks learn"]


def test_tokenizer_invalid_ngrams():
    tokens = ["word"]
    assert Tokenizer.get_ngrams(tokens, n=2) == []
    assert Tokenizer.get_ngrams(tokens, n=0) == []

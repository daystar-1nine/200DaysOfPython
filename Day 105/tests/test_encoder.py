"""
Unit tests for TextEncoder and Tokenizer.
Day 105: Neural NLP & Text Classification.
"""

import pytest
from app.preprocessing.tokenizer import Tokenizer
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.encoder import TextEncoder


@pytest.fixture
def fitted_encoder():
    tokenizer = Tokenizer()
    corpus = [["fast", "api", "framework"], ["neural", "nlp", "model"]]
    vocab = Vocabulary().fit(corpus)
    return TextEncoder(tokenizer, vocab)


def test_tokenizer_clean_words():
    t = Tokenizer()
    tokens = t.tokenize("Hello, World! This is a TEST.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_tokenizer_empty_and_none():
    t = Tokenizer()
    assert t.tokenize("") == []
    assert t.tokenize("   ") == []
    assert t.tokenize(None) == []


def test_text_encoder_encode_text(fitted_encoder):
    encoded = fitted_encoder.encode_text("fast neural model")
    assert len(encoded) == 3
    assert all(isinstance(x, int) for x in encoded)


def test_text_encoder_encode_corpus(fitted_encoder):
    corpus = ["fast api", "neural model"]
    encoded = fitted_encoder.encode_corpus(corpus)
    assert len(encoded) == 2
    assert len(encoded[0]) == 2
    assert len(encoded[1]) == 2


def test_text_encoder_decode_ids(fitted_encoder):
    encoded = fitted_encoder.encode_text("fast api")
    decoded = fitted_encoder.decode_ids(encoded)
    assert decoded == "fast api"

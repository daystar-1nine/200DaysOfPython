"""Tests for vocabulary and bidirectional mapping."""
import pytest
from app.data.vocabulary import Vocabulary

def test_vocabulary_build():
    docs = [["cat", "dog"], ["cat", "fish"]]
    vocab = Vocabulary().build_vocab(docs)
    assert len(vocab) == 3
    assert "cat" in vocab and "dog" in vocab and "fish" in vocab

def test_vocabulary_bidirectional_mapping():
    docs = [["apple", "banana"]]
    vocab = Vocabulary().build_vocab(docs)
    a_id = vocab.get_id("apple")
    assert vocab.get_word(a_id) == "apple"
    assert vocab.get_id("unknown") is None

def test_vocabulary_min_count():
    docs = [["cat", "dog", "dog", "dog"]]
    vocab = Vocabulary(min_count=2).build_vocab(docs)
    assert "dog" in vocab
    assert "cat" not in vocab

def test_vocabulary_frequencies():
    docs = [["a", "b", "a", "c", "a", "b"]]
    vocab = Vocabulary().build_vocab(docs)
    dist = vocab.get_frequency_distribution()
    assert dist["a"] == 3
    assert dist["b"] == 2
    assert dist["c"] == 1

def test_vocabulary_contains():
    vocab = Vocabulary().build_vocab([["python"]])
    assert "python" in vocab
    assert "java" not in vocab

def test_vocabulary_empty():
    vocab = Vocabulary().build_vocab([])
    assert len(vocab) == 0

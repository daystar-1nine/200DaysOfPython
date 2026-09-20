"""Tests for context pair generation."""
import pytest
from app.data.vocabulary import Vocabulary
from app.preprocessing.context_pairs import generate_context_pairs, generate_all_pairs

def test_context_pairs_window_1():
    vocab = Vocabulary().build_vocab([["a", "b", "c"]])
    pairs = generate_context_pairs(["a", "b", "c"], vocab, window_size=1)
    # (a, b), (b, a), (b, c), (c, b)
    assert len(pairs) == 4
    id_a = vocab.get_id("a")
    id_b = vocab.get_id("b")
    assert (id_a, id_b) in pairs
    assert (id_b, id_a) in pairs

def test_context_pairs_window_2():
    vocab = Vocabulary().build_vocab([["a", "b", "c", "d"]])
    pairs = generate_context_pairs(["a", "b", "c", "d"], vocab, window_size=2)
    # a connects to b, c
    id_a = vocab.get_id("a")
    id_c = vocab.get_id("c")
    assert (id_a, id_c) in pairs

def test_context_pairs_short_doc():
    vocab = Vocabulary().build_vocab([["a"]])
    pairs = generate_context_pairs(["a"], vocab, window_size=2)
    assert pairs == []

def test_context_pairs_invalid_window():
    vocab = Vocabulary().build_vocab([["a", "b"]])
    with pytest.raises(ValueError):
        generate_context_pairs(["a", "b"], vocab, window_size=0)

def test_generate_all_pairs():
    corpus = [["a", "b"], ["c", "d"]]
    vocab = Vocabulary().build_vocab(corpus)
    pairs = generate_all_pairs(corpus, vocab, window_size=1)
    assert len(pairs) == 4

def test_context_pairs_boundary():
    vocab = Vocabulary().build_vocab([["a", "b", "c"]])
    pairs = generate_context_pairs(["a", "b", "c"], vocab, window_size=5)
    # Every pair except self
    assert len(pairs) == 6

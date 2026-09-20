"""Tests for negative sampling."""
import pytest
from app.data.vocabulary import Vocabulary
from app.embeddings.negative_sampling import NegativeSampler

def test_sampler_count():
    vocab = Vocabulary().build_vocab([["cat", "dog", "fish", "bird", "lion", "tiger"]])
    sampler = NegativeSampler(vocab, seed=42)
    negs = sampler.sample(num_samples=3, target_id=0, context_id=1)
    assert len(negs) == 3

def test_sampler_excludes_target_and_context():
    vocab = Vocabulary().build_vocab([["a", "b", "c", "d", "e"]])
    sampler = NegativeSampler(vocab, seed=42)
    for _ in range(20):
        negs = sampler.sample(num_samples=2, target_id=0, context_id=1)
        assert 0 not in negs
        assert 1 not in negs

def test_sampler_table_structure():
    vocab = Vocabulary().build_vocab([["a", "b"]])
    sampler = NegativeSampler(vocab, table_size=1000)
    assert len(sampler.table) > 0

def test_sampler_deterministic_seed():
    vocab = Vocabulary().build_vocab([["a", "b", "c", "d", "e"]])
    s1 = NegativeSampler(vocab, seed=123)
    s2 = NegativeSampler(vocab, seed=123)
    assert s1.sample(3, 0, 1) == s2.sample(3, 0, 1)

def test_sampler_unigram_power():
    # Frequent words should appear more in lookup table
    vocab = Vocabulary().build_vocab([["a"] * 100 + ["b"] * 1 + ["c", "d"]])
    sampler = NegativeSampler(vocab, power=0.75, table_size=10000)
    id_a = vocab.get_id("a")
    id_b = vocab.get_id("b")
    count_a = sum(sampler.table == id_a)
    count_b = sum(sampler.table == id_b)
    assert count_a > count_b

def test_sampler_sample_zero():
    vocab = Vocabulary().build_vocab([["a", "b", "c"]])
    sampler = NegativeSampler(vocab)
    assert sampler.sample(0, 0, 1) == []

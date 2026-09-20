"""Tests for complete Word2Vec model and training."""
import pytest
import numpy as np
from app.embeddings.word2vec import Word2Vec
from app.embeddings.initialization import initialize_embeddings

def test_word2vec_initialization():
    W_in, W_out = initialize_embeddings(vocab_size=10, embedding_dim=8, seed=42)
    assert W_in.shape == (10, 8)
    assert W_out.shape == (10, 8)

def test_word2vec_training_reduces_loss():
    corpus = [
        ["the", "cat", "drinks", "milk"],
        ["the", "dog", "drinks", "water"],
        ["the", "cat", "likes", "fish"]
    ]
    model = Word2Vec(embedding_dim=8, epochs=15, learning_rate=0.05, seed=42)
    model.fit(corpus)
    assert len(model.epoch_losses) == 15
    assert model.epoch_losses[-1] < model.epoch_losses[0]

def test_word2vec_get_vector():
    corpus = [["hello", "world"]]
    model = Word2Vec(embedding_dim=4, epochs=2).fit(corpus)
    vec = model.get_vector("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (4,)
    assert model.get_vector("unknown") is None

def test_word2vec_unfitted_raises():
    model = Word2Vec()
    with pytest.raises(ValueError):
        model.get_vector("test")

def test_word2vec_most_similar():
    corpus = [
        ["the", "cat", "drinks", "milk"],
        ["the", "dog", "drinks", "milk"],
        ["the", "kitten", "drinks", "milk"]
    ]
    model = Word2Vec(embedding_dim=8, epochs=10, seed=42).fit(corpus)
    similar = model.most_similar("cat", top_k=2)
    assert len(similar) <= 2
    for word, score in similar:
        assert isinstance(word, str)
        assert -1.0 <= score <= 1.0

def test_word2vec_too_small_corpus_raises():
    model = Word2Vec()
    with pytest.raises(ValueError):
        model.fit([["single"]])

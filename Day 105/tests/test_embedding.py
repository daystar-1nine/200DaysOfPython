"""
Unit tests for embedding lookup and visualizer.
Day 105: Neural NLP & Text Classification.
"""

import numpy as np
import pytest
from app.embeddings.lookup import embedding_lookup
from app.embeddings.visualization import EmbeddingVisualizer


def test_embedding_lookup_single_sequence():
    W = np.arange(12, dtype=np.float32).reshape(4, 3)
    ids = [0, 2]
    out = embedding_lookup(W, ids)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], W[0])
    assert np.allclose(out[1], W[2])


def test_embedding_lookup_batch():
    W = np.random.randn(10, 5).astype(np.float32)
    batch_ids = [[1, 2, 3], [4, 5, 6]]
    out = embedding_lookup(W, batch_ids)
    assert out.shape == (2, 3, 5)


def test_embedding_lookup_out_of_bounds():
    W = np.ones((5, 3))
    with pytest.raises(IndexError):
        embedding_lookup(W, [5])  # valid indices are 0..4
    with pytest.raises(IndexError):
        embedding_lookup(W, [-1])


def test_embedding_lookup_invalid_matrix():
    with pytest.raises(ValueError):
        embedding_lookup(np.ones(5), [1, 2])


def test_embedding_visualizer_project_pca():
    W = np.random.randn(8, 16).astype(np.float32)
    words = [f"w_{i}" for i in range(8)]
    pca_df = EmbeddingVisualizer.project_pca(W, words, n_components=2)
    assert len(pca_df) == 8
    assert "word" in pca_df.columns
    assert "pc1" in pca_df.columns
    assert "pc2" in pca_df.columns

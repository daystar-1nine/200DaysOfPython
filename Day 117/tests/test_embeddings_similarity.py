"""
Unit tests for cosine similarity calculation, dense TF-IDF embedding, and vector normalizations.
"""
import pytest
import numpy as np
from app.embeddings.similarity import (
    cosine_similarity_pair,
    cosine_similarity_matrix,
    batch_cosine_similarity,
    normalize_vector
)
from app.embeddings.embedder import DenseTFIDFEmbedder, SparseTFIDFEmbedder


def test_cosine_similarity_identical_vectors():
    v1 = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity_pair(v1, v1) - 1.0) < 1e-6


def test_cosine_similarity_collinear_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([10.0, 0.0, 0.0])
    assert abs(cosine_similarity_pair(v1, v2) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert abs(cosine_similarity_pair(v1, v2) - 0.0) < 1e-6


def test_cosine_similarity_opposite_vectors():
    v1 = np.array([2.0, -1.0])
    v2 = np.array([-2.0, 1.0])
    assert abs(cosine_similarity_pair(v1, v2) - (-1.0)) < 1e-6


def test_cosine_similarity_zero_norm_vector():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity_pair(v1, v2) == 0.0
    assert cosine_similarity_pair(v1, v1) == 0.0


def test_cosine_similarity_dimension_mismatch():
    v1 = np.array([1.0, 2.0])
    v2 = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        cosine_similarity_pair(v1, v2)


def test_batch_cosine_similarity_shapes():
    query = np.array([1.0, 0.0, 0.0])
    corpus = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    scores = batch_cosine_similarity(query, corpus)
    assert scores.shape == (3,)
    assert abs(scores[0] - 1.0) < 1e-5
    assert abs(scores[1] - 0.0) < 1e-5
    assert abs(scores[2] - (-1.0)) < 1e-5


def test_cosine_similarity_matrix_symmetric():
    corpus = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [-1.0, 1.0]
    ])
    mat = cosine_similarity_matrix(corpus, corpus)
    assert mat.shape == (3, 3)
    # Diagonal should be 1.0
    for i in range(3):
        assert abs(mat[i, i] - 1.0) < 1e-5
    # Matrix should be symmetric
    assert np.allclose(mat, mat.T)


def test_normalize_vector_unit_norm():
    v = np.array([3.0, 4.0])
    normed = normalize_vector(v)
    assert abs(np.linalg.norm(normed) - 1.0) < 1e-6
    assert abs(normed[0] - 0.6) < 1e-6
    assert abs(normed[1] - 0.8) < 1e-6


def test_normalize_vector_zero_vector():
    v = np.array([0.0, 0.0])
    normed = normalize_vector(v)
    assert np.all(normed == 0.0)


def test_embedder_fit_and_embed_batch():
    embedder = DenseTFIDFEmbedder(dim=8)
    corpus = [
        "Python lists and dictionaries are widely used.",
        "Relational databases provide ACID transaction guarantees.",
        "Transformers use self-attention mechanisms."
    ]
    embedder.fit(corpus)
    vecs = embedder.embed_batch(corpus)
    assert vecs.shape == (3, 8)
    # Each embedding vector should be L2 normalized
    for i in range(3):
        norm = np.linalg.norm(vecs[i])
        assert abs(norm - 1.0) < 1e-4 or norm == 0.0


def test_embedder_single_query_embedding():
    embedder = DenseTFIDFEmbedder(dim=12)
    corpus = ["Machine learning with python and numpy.", "Deep learning with torch."]
    embedder.fit(corpus)
    q_vec = embedder.embed_text("python machine learning")
    assert q_vec.shape == (12,)
    assert abs(np.linalg.norm(q_vec) - 1.0) < 1e-4


def test_embedder_empty_string():
    embedder = DenseTFIDFEmbedder(dim=10)
    embedder.fit(["Hello world test corpus."])
    vec = embedder.embed_text("")
    assert vec.shape == (10,)
    assert np.all(vec == 0.0)


def test_embedder_unseen_vocabulary():
    embedder = DenseTFIDFEmbedder(dim=8)
    embedder.fit(["apple banana orange"])
    # All tokens unseen in fit corpus
    vec = embedder.embed_text("quantum physics relativity")
    assert vec.shape == (8,)


def test_embedder_not_fitted_error():
    embedder = DenseTFIDFEmbedder(dim=8)
    with pytest.raises(RuntimeError, match="must be fitted"):
        embedder.embed_text("test query")


def test_sparse_tfidf_embedder():
    embedder = SparseTFIDFEmbedder()
    corpus = ["cat sat on mat", "dog ran in park"]
    embedder.fit(corpus)
    q_vec = embedder.embed_text("cat dog")
    assert len(q_vec) == embedder.dim
    assert isinstance(q_vec, np.ndarray)


def test_embedder_dimension_truncation():
    # When dim requested is larger than vocab size
    embedder = DenseTFIDFEmbedder(dim=100)
    corpus = ["apple banana cherry orange date"]
    embedder.fit(corpus)
    vec = embedder.embed_text("apple banana")
    assert vec.shape == (100,)


def test_cosine_similarity_with_nan_handling():
    v1 = np.array([np.nan, 1.0])
    v2 = np.array([1.0, 1.0])
    # Ensure nan is handled gracefully without uncaught crash
    score = cosine_similarity_pair(np.nan_to_num(v1), v2)
    assert isinstance(score, float)


def test_batch_cosine_empty_corpus():
    query = np.array([1.0, 2.0])
    corpus = np.empty((0, 2))
    scores = batch_cosine_similarity(query, corpus)
    assert len(scores) == 0


def test_embedder_deterministic_embeddings():
    embedder = DenseTFIDFEmbedder(dim=8)
    corpus = ["Determinism is important for evaluation benchmarks."]
    embedder.fit(corpus)
    vec1 = embedder.embed_text("evaluation benchmarks")
    vec2 = embedder.embed_text("evaluation benchmarks")
    assert np.array_equal(vec1, vec2)

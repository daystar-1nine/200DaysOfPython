"""
Unit tests for cosine similarity and vectorized similarity computation.
"""

import numpy as np
import pytest
from app.retrieval.ranking import (
    cosine_similarity,
    compute_similarities,
    normalize_scores
)


def test_cosine_similarity_identical():
    v = np.array([0.5, 0.5, 0.5])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(v1, v2)) < 1e-6


def test_cosine_similarity_opposite():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([-1.0, -2.0, -3.0])
    assert abs(cosine_similarity(v1, v2) - (-1.0)) < 1e-6


def test_cosine_similarity_zero_vector():
    v1 = np.array([1.0, 2.0, 3.0])
    v0 = np.array([0.0, 0.0, 0.0])
    assert cosine_similarity(v1, v0) == 0.0
    assert cosine_similarity(v0, v0) == 0.0


def test_cosine_similarity_shape_mismatch():
    v1 = np.array([1.0, 2.0])
    v2 = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        cosine_similarity(v1, v2)


def test_compute_similarities_vectorized():
    query = np.array([1.0, 0.0])
    docs = np.array([
        [1.0, 0.0],   # identical -> 1.0
        [0.0, 1.0],   # orthogonal -> 0.0
        [-1.0, 0.0],  # opposite -> -1.0
        [0.0, 0.0]    # zero -> 0.0
    ])
    scores = compute_similarities(query, docs)
    assert scores.shape == (4,)
    assert abs(scores[0] - 1.0) < 1e-6
    assert abs(scores[1] - 0.0) < 1e-6
    assert abs(scores[2] - (-1.0)) < 1e-6
    assert abs(scores[3] - 0.0) < 1e-6


def test_compute_similarities_zero_query():
    query = np.array([0.0, 0.0])
    docs = np.array([[1.0, 2.0], [3.0, 4.0]])
    scores = compute_similarities(query, docs)
    assert np.allclose(scores, np.zeros(2))


def test_normalize_scores_standard():
    scores = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    norm = normalize_scores(scores)
    assert abs(np.min(norm) - 0.0) < 1e-6
    assert abs(np.max(norm) - 1.0) < 1e-6
    assert np.all((norm >= 0.0) & (norm <= 1.0))


def test_normalize_scores_constant():
    scores = np.array([0.5, 0.5, 0.5])
    norm = normalize_scores(scores)
    assert np.allclose(norm, np.ones(3))

    negative_constant = np.array([-0.5, -0.5])
    norm_neg = normalize_scores(negative_constant)
    assert np.allclose(norm_neg, np.zeros(2))

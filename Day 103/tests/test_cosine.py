"""Tests for cosine similarity."""
import pytest
import numpy as np
from app.similarity.cosine import cosine_similarity

def test_cosine_same_direction():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == 1.0

def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == 0.0

def test_cosine_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == -1.0

def test_cosine_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0

def test_cosine_identical():
    a = np.array([0.5, -0.5, 0.25])
    assert pytest.approx(cosine_similarity(a, a), abs=1e-6) == 1.0

def test_cosine_bounded():
    rng = np.random.default_rng(42)
    for _ in range(10):
        a = rng.normal(size=5)
        b = rng.normal(size=5)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

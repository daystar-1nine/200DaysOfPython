"""
Unit tests for Precision@K metric.
"""

from app.evaluation.precision_at_k import precision_at_k


def test_precision_at_k_basic():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5]
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    assert precision_at_k(retrieved, relevant, k=3) == 2.0 / 3.0
    assert precision_at_k(retrieved, relevant, k=5) == 3.0 / 5.0


def test_precision_at_k_zero_or_negative_k():
    assert precision_at_k([1, 2], [1], k=0) == 0.0
    assert precision_at_k([1, 2], [1], k=-2) == 0.0


def test_precision_at_k_empty_inputs():
    assert precision_at_k([], [1, 2], k=3) == 0.0
    assert precision_at_k([1, 2], [], k=3) == 0.0


def test_precision_at_k_all_relevant():
    retrieved = [1, 2, 3]
    relevant = [1, 2, 3, 4, 5]
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_none_relevant():
    retrieved = [1, 2, 3]
    relevant = [4, 5, 6]
    assert precision_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_at_k_large_k():
    retrieved = [1, 2]
    relevant = [1]
    # k=5, only 1 hit out of k=5
    assert precision_at_k(retrieved, relevant, k=5) == 1.0 / 5.0

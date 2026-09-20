"""
Unit tests for Recall@K metric.
"""

from app.evaluation.recall_at_k import recall_at_k


def test_recall_at_k_basic():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 6, 7]  # 4 relevant in total
    assert recall_at_k(retrieved, relevant, k=1) == 0.25  # hit 1
    assert recall_at_k(retrieved, relevant, k=3) == 0.50  # hits 1, 3
    assert recall_at_k(retrieved, relevant, k=5) == 0.50  # hits 1, 3


def test_recall_at_k_zero_or_negative_k():
    assert recall_at_k([1, 2], [1], k=0) == 0.0
    assert recall_at_k([1, 2], [1], k=-1) == 0.0


def test_recall_at_k_empty_inputs():
    assert recall_at_k([], [1, 2], k=5) == 0.0
    assert recall_at_k([1, 2], [], k=5) == 0.0


def test_recall_at_k_perfect():
    retrieved = [1, 2, 3]
    relevant = [1, 2]
    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_none_found():
    retrieved = [10, 20, 30]
    relevant = [1, 2, 3]
    assert recall_at_k(retrieved, relevant, k=3) == 0.0

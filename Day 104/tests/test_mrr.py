"""
Unit tests for Reciprocal Rank and MRR metrics.
"""

from app.evaluation.mrr import reciprocal_rank, mean_reciprocal_rank


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([1, 2, 3], [1]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank([1, 2, 3], [2]) == 0.5


def test_reciprocal_rank_third_position():
    assert abs(reciprocal_rank([1, 2, 3], [3]) - (1.0 / 3.0)) < 1e-6


def test_reciprocal_rank_not_found():
    assert reciprocal_rank([1, 2, 3], [99]) == 0.0


def test_reciprocal_rank_empty():
    assert reciprocal_rank([], [1]) == 0.0
    assert reciprocal_rank([1], []) == 0.0


def test_mean_reciprocal_rank():
    scores = [1.0, 0.5, 0.25, 0.0]
    expected = (1.0 + 0.5 + 0.25 + 0.0) / 4.0
    assert abs(mean_reciprocal_rank(scores) - expected) < 1e-6


def test_mean_reciprocal_rank_empty():
    assert mean_reciprocal_rank([]) == 0.0

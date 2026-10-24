"""
Unit tests for information retrieval metrics: Precision@K, Recall@K, MRR, and HitRate@K.
"""
import pytest
from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank
)


def test_precision_at_k_perfect():
    ret = ["doc1", "doc2", "doc3"]
    rel = ["doc1", "doc2", "doc3"]
    assert precision_at_k(ret, rel, k=1) == 1.0
    assert precision_at_k(ret, rel, k=3) == 1.0


def test_precision_at_k_partial_hits():
    ret = ["doc1", "doc2", "doc3", "doc4"]
    rel = ["doc1", "doc3"]
    # k=2 -> ["doc1", "doc2"] -> only doc1 matches -> 1/2 = 0.5
    assert precision_at_k(ret, rel, k=2) == 0.5
    # k=4 -> ["doc1", "doc2", "doc3", "doc4"] -> doc1, doc3 match -> 2/4 = 0.5
    assert precision_at_k(ret, rel, k=4) == 0.5


def test_precision_at_k_zero_hits():
    ret = ["docA", "docB"]
    rel = ["docX", "docY"]
    assert precision_at_k(ret, rel, k=1) == 0.0
    assert precision_at_k(ret, rel, k=2) == 0.0


def test_precision_at_k_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["doc1"], ["doc1"], k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["doc1"], ["doc1"], k=-2)


def test_recall_at_k_perfect():
    ret = ["doc1", "doc2", "doc3"]
    rel = ["doc1", "doc2"]
    # Both relevant docs in top 2 -> 2/2 = 1.0
    assert recall_at_k(ret, rel, k=2) == 1.0
    assert recall_at_k(ret, rel, k=3) == 1.0


def test_recall_at_k_partial():
    ret = ["doc1", "doc2", "doc3"]
    rel = ["doc1", "doc4"]
    # Only doc1 is retrieved in top 3 -> 1/2 = 0.5
    assert recall_at_k(ret, rel, k=3) == 0.5


def test_recall_at_k_zero():
    ret = ["docA", "docB"]
    rel = ["docC"]
    assert recall_at_k(ret, rel, k=2) == 0.0


def test_recall_at_k_empty_relevant():
    # If no relevant items expected (e.g. out of domain)
    assert recall_at_k([], [], k=3) == 1.0
    assert recall_at_k(["doc1"], [], k=3) == 0.0


def test_hit_rate_at_k_binary():
    ret = ["docA", "docB", "docC"]
    assert hit_rate_at_k(ret, ["docB"], k=1) == 0.0
    assert hit_rate_at_k(ret, ["docB"], k=2) == 1.0
    assert hit_rate_at_k(ret, ["docZ"], k=3) == 0.0


def test_reciprocal_rank_first_rank():
    assert reciprocal_rank(["doc1", "doc2"], ["doc1"]) == 1.0


def test_reciprocal_rank_second_rank():
    assert reciprocal_rank(["docA", "docB", "docC"], ["docB"]) == 0.5


def test_reciprocal_rank_third_rank():
    assert abs(reciprocal_rank(["docA", "docB", "docC"], ["docC"]) - (1.0 / 3.0)) < 1e-6


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["docA", "docB"], ["docZ"]) == 0.0


def test_mean_reciprocal_rank_workload():
    all_ret = [
        ["doc1", "doc2"],  # hit at rank 1 -> 1.0
        ["docA", "docB"],  # hit at rank 2 -> 0.5
        ["docX", "docY"]   # hit at rank 0 -> 0.0
    ]
    all_rel = [["doc1"], ["docB"], ["docZ"]]
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert abs(mean_reciprocal_rank(all_ret, all_rel) - 0.5) < 1e-6


def test_mean_reciprocal_rank_length_mismatch():
    with pytest.raises(ValueError, match="Lengths.*must match"):
        mean_reciprocal_rank([["doc1"]], [["doc1"], ["doc2"]])


def test_mean_reciprocal_rank_empty():
    assert mean_reciprocal_rank([], []) == 0.0

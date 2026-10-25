"""
Unit tests for Information Retrieval metrics: Precision@K, Recall@K, Hit@K, MRR, DCG@K, and nDCG@K.
"""
import pytest
import math
from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    discounted_cumulative_gain_at_k,
    ndcg_at_k,
    evaluate_ranking,
)


def test_precision_at_k_bounds():
    with pytest.raises(ValueError):
        precision_at_k(["d1"], {"d1"}, k=0)
    with pytest.raises(ValueError):
        precision_at_k(["d1"], {"d1"}, k=-1)


def test_precision_at_k_calculation():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3"}

    assert precision_at_k(retrieved, relevant, k=1) == 1.0  # d1 is in relevant
    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # d1 relevant, d2 irrelevant
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # d1, d3 relevant / 4 = 2/4 = 0.5
    assert precision_at_k(retrieved, relevant, k=5) == 0.4  # 2 / 5 = 0.4


def test_recall_at_k_calculation():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3", "d6", "d7"}  # 4 total relevant

    assert recall_at_k(retrieved, relevant, k=1) == 0.25  # 1 / 4
    assert recall_at_k(retrieved, relevant, k=2) == 0.25  # 1 / 4
    assert recall_at_k(retrieved, relevant, k=3) == 0.50  # 2 / 4 (d1, d3)
    assert recall_at_k(retrieved, relevant, k=5) == 0.50


def test_recall_at_k_capped_at_one():
    # If same document is retrieved multiple times in chunks, recall cannot exceed 1.0
    retrieved = ["d1", "d1", "d1"]
    relevant = {"d1"}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k():
    retrieved = ["d_wrong_1", "d_wrong_2", "d_target", "d_wrong_3"]
    relevant = {"d_target"}

    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, k=5) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["d1", "d2", "d3"], {"d1"}) == 1.0
    assert reciprocal_rank(["d1", "d2", "d3"], {"d2"}) == 0.5
    assert reciprocal_rank(["d1", "d2", "d3"], {"d3"}) == pytest.approx(1.0 / 3.0)
    assert reciprocal_rank(["d1", "d2", "d3"], {"d_missing"}) == 0.0
    assert reciprocal_rank([], {"d1"}) == 0.0


def test_dcg_at_k_calculation():
    # Rank 1: rel=3 -> (2^3 - 1) / log2(2) = 7 / 1 = 7.0
    # Rank 2: rel=2 -> (2^2 - 1) / log2(3) = 3 / 1.58496 = 1.89278
    # Total DCG@2 = 7.0 + 1.89278 = 8.8928
    retrieved = ["d1", "d2"]
    grades = {"d1": 3.0, "d2": 2.0}
    dcg = discounted_cumulative_gain_at_k(retrieved, grades, k=2)
    expected = 7.0 + (3.0 / math.log2(3.0))
    assert dcg == pytest.approx(expected, rel=1e-4)


def test_ndcg_at_k_perfect_ranking():
    # Ideal ranking: items ordered strictly by relevance
    retrieved = ["d_best", "d_mid", "d_low"]
    grades = {"d_best": 3.0, "d_mid": 2.0, "d_low": 1.0}
    ndcg = ndcg_at_k(retrieved, grades, k=3)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_at_k_inverted_ranking():
    retrieved = ["d_low", "d_mid", "d_best"]
    grades = {"d_best": 3.0, "d_mid": 2.0, "d_low": 1.0}
    ndcg = ndcg_at_k(retrieved, grades, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_at_k_no_relevant():
    retrieved = ["d_unrelated"]
    grades = {"d_other": 3.0}
    assert ndcg_at_k(retrieved, grades, k=1) == 0.0


def test_evaluate_ranking_full_suite():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3"}
    grades = {"d1": 3.0, "d3": 2.0}
    res = evaluate_ranking(retrieved, relevant, grades, k_values=[1, 3, 5])
    
    assert "mrr" in res
    assert "p@1" in res and "p@3" in res and "p@5" in res
    assert "r@1" in res and "r@3" in res and "r@5" in res
    assert "ndcg@1" in res and "ndcg@3" in res and "ndcg@5" in res
    assert res["mrr"] == 1.0

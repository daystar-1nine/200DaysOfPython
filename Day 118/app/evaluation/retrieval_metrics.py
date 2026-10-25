"""
Retrieval Evaluation Metrics: Precision@K, Recall@K, Hit Rate@K, MRR, DCG@K, and nDCG@K.
Implements exact information retrieval evaluation standards with graded relevance.
"""
from typing import List, Dict, Set, Sequence, Optional, Union
import math
import numpy as np

from app.index.vector_index import ScoredChunk


def precision_at_k(
    retrieved_chunk_or_doc_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Computes Precision@K:
        Precision@K = (Number of unique relevant items in top-K) / K
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_chunk_or_doc_ids[:k]
    if not top_k_ids:
        return 0.0

    # Credit unique relevant items in top-k
    seen = set()
    unique_top_k = []
    for item in top_k_ids:
        if item not in seen:
            seen.add(item)
            unique_top_k.append(item)

    relevant_count = sum(1 for item in unique_top_k if item in relevant_ids)
    return relevant_count / float(k)


def recall_at_k(
    retrieved_chunk_or_doc_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Computes Recall@K:
        Recall@K = (Number of unique relevant items in top-K) / (Total relevant items)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_chunk_or_doc_ids[:k]
    matched_relevant_count = len(set(top_k_ids).intersection(relevant_ids))
    return min(1.0, matched_relevant_count / float(len(relevant_ids)))


def hit_rate_at_k(
    retrieved_chunk_or_doc_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Computes Hit Rate@K (1.0 if at least one relevant item in top-K, else 0.0).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_chunk_or_doc_ids[:k]
    return 1.0 if any(item in relevant_ids for item in top_k_ids) else 0.0


def reciprocal_rank(
    retrieved_chunk_or_doc_ids: Sequence[str],
    relevant_ids: Set[str]
) -> float:
    """
    Computes Reciprocal Rank (RR):
        RR = 1 / rank_first_relevant (1-based), or 0.0 if not found.
    """
    if not relevant_ids:
        return 0.0

    for rank_1based, item_id in enumerate(retrieved_chunk_or_doc_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank_1based

    return 0.0


def discounted_cumulative_gain_at_k(
    retrieved_chunk_or_doc_ids: Sequence[str],
    graded_relevance: Dict[str, float],
    k: int
) -> float:
    """
    Computes DCG@K with exponential gain:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    Only credits each document once at its highest rank (subsequent duplicate chunks receive rel=0).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    top_k_ids = retrieved_chunk_or_doc_ids[:k]
    dcg = 0.0
    seen_docs = set()
    for i, item_id in enumerate(top_k_ids, start=1):
        if item_id in seen_docs:
            rel = 0.0
        else:
            rel = graded_relevance.get(item_id, 0.0)
            seen_docs.add(item_id)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 1.0)
        dcg += gain / discount

    return dcg


def ndcg_at_k(
    retrieved_chunk_or_doc_ids: Sequence[str],
    graded_relevance: Dict[str, float],
    k: int
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@K):
        nDCG@K = DCG@K / IDCG@K
    where IDCG@K is the ideal DCG obtained by sorting true relevance grades descending.
    Bounded strictly in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not graded_relevance:
        return 0.0

    actual_dcg = discounted_cumulative_gain_at_k(retrieved_chunk_or_doc_ids, graded_relevance, k)

    # Compute Ideal DCG (IDCG)
    sorted_grades = sorted(graded_relevance.values(), reverse=True)
    ideal_top_k = sorted_grades[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_top_k, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 1.0)
        idcg += gain / discount

    if idcg == 0.0:
        return 0.0

    return min(1.0, actual_dcg / idcg)


def evaluate_ranking(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
    graded_relevance: Optional[Dict[str, float]] = None,
    k_values: Sequence[int] = (1, 3, 5, 10)
) -> Dict[str, float]:
    """
    Computes full metric suite for a single query ranking.
    """
    grades = graded_relevance or {d: 1.0 for d in relevant_doc_ids}
    metrics = {
        "mrr": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
    }

    for k in k_values:
        metrics[f"p@{k}"] = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"r@{k}"] = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"hit@{k}"] = hit_rate_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_doc_ids, grades, k)

    return metrics

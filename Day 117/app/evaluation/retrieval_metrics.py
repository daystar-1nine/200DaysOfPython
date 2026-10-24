"""
Retrieval evaluation metrics: Precision@K, Recall@K, MRR, HitRate@K, and F1@K.
"""
from typing import List, Set, Union


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Computes Precision@K:
        Precision@K = |Retrieved[:k] intersect Relevant| / k
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not retrieved_ids or not relevant_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_k & relevant_set)
    return float(hits / k)


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Computes Recall@K:
        Recall@K = |Retrieved[:k] intersect Relevant| / |Relevant|
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0
    if not retrieved_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_k & relevant_set)
    return float(hits / len(relevant_set))


def hit_rate_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Computes HitRate@K (Binary indicator if at least one relevant document is in top k).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not retrieved_ids or not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if bool(retrieved_k & relevant_set) else 0.0


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Computes Reciprocal Rank (RR):
        RR = 1 / rank_first_relevant (1-indexed), or 0.0 if not found.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return float(1.0 / rank)
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across all evaluated queries.
    """
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("Lengths of all_retrieved and all_relevant must match.")
    if not all_retrieved:
        return 0.0

    total_rr = sum(
        reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    )
    return float(total_rr / len(all_retrieved))


def f1_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Computes harmonic mean F1 score at rank k.
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))

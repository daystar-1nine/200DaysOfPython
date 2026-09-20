"""
Precision@K metric for Day 104 Semantic Search Engine.
"""

from typing import Any, List, Set


def precision_at_k(
    retrieved_ids: List[Any],
    relevant_ids: List[Any],
    k: int
) -> float:
    """Calculate Precision@K: fraction of retrieved documents in top K that are relevant.
    
    Formula:
        Precision@K = |Retrieved_K ∩ Relevant| / K
        
    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List or set of ground-truth relevant document IDs.
        k: Number of top results to evaluate.
        
    Returns:
        Precision@K float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k_retrieved = retrieved_ids[:k]
    relevant_set: Set[Any] = set(relevant_ids)

    relevant_retrieved_count = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    return float(relevant_retrieved_count / k)

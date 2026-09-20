"""
Recall@K metric for Day 104 Semantic Search Engine.
"""

from typing import Any, List, Set


def recall_at_k(
    retrieved_ids: List[Any],
    relevant_ids: List[Any],
    k: int
) -> float:
    """Calculate Recall@K: fraction of total relevant documents retrieved in top K.
    
    Formula:
        Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|
        
    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List or set of ground-truth relevant document IDs.
        k: Number of top results to evaluate.
        
    Returns:
        Recall@K float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set: Set[Any] = set(relevant_ids)
    if len(relevant_set) == 0:
        return 0.0

    top_k_retrieved = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)

    return float(hits / len(relevant_set))

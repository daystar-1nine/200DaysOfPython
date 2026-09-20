"""
Reciprocal Rank and Mean Reciprocal Rank (MRR) metrics for Day 104 Semantic Search Engine.
"""

from typing import Any, List, Set


def reciprocal_rank(
    retrieved_ids: List[Any],
    relevant_ids: List[Any]
) -> float:
    """Calculate the Reciprocal Rank (RR) for a single query.
    
    Formula:
        RR = 1 / rank_first_relevant
        (where rank is 1-indexed; returns 0.0 if no relevant document is retrieved)
        
    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List or set of ground-truth relevant document IDs.
        
    Returns:
        Reciprocal Rank float in [0.0, 1.0].
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set: Set[Any] = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return float(1.0 / rank)

    return 0.0


def mean_reciprocal_rank(rr_scores: List[float]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across a list of query reciprocal ranks.
    
    Formula:
        MRR = (1 / |Q|) * sum_{q in Q} RR_q
        
    Args:
        rr_scores: List of individual query reciprocal rank values.
        
    Returns:
        Mean Reciprocal Rank float in [0.0, 1.0].
    """
    if not rr_scores:
        return 0.0

    return float(sum(rr_scores) / len(rr_scores))

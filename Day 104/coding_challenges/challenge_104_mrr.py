"""
Coding Challenge 104.7 & 104.8: Reciprocal Rank and Mean Reciprocal Rank (MRR).
"""

from typing import Any, List


def reciprocal_rank(retrieved_ids: List[Any], relevant_ids: List[Any]) -> float:
    """Challenge 7: Implement Reciprocal Rank (RR).
    
    Formula: 1 / rank_of_first_relevant_doc
    (1-indexed; 0.0 if none retrieved)
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel_set:
            return float(1.0 / rank)
    return 0.0


def mean_reciprocal_rank(rr_scores: List[float]) -> float:
    """Challenge 8: Implement Mean Reciprocal Rank (MRR).
    
    Formula: (1 / |Q|) * sum_{q in Q} RR_q
    """
    if not rr_scores:
        return 0.0
    return float(sum(rr_scores) / len(rr_scores))


if __name__ == "__main__":
    q1_retrieved = [10, 20, 30]
    q1_relevant = [10]  # rank 1 -> RR = 1.0

    q2_retrieved = [10, 20, 30]
    q2_relevant = [20]  # rank 2 -> RR = 0.5

    q3_retrieved = [10, 20, 30]
    q3_relevant = [30]  # rank 3 -> RR = 1/3

    q4_retrieved = [10, 20, 30]
    q4_relevant = [99]  # not found -> RR = 0.0

    rr1 = reciprocal_rank(q1_retrieved, q1_relevant)
    rr2 = reciprocal_rank(q2_retrieved, q2_relevant)
    rr3 = reciprocal_rank(q3_retrieved, q3_relevant)
    rr4 = reciprocal_rank(q4_retrieved, q4_relevant)

    assert rr1 == 1.0
    assert rr2 == 0.5
    assert abs(rr3 - 1/3) < 1e-5
    assert rr4 == 0.0

    mrr = mean_reciprocal_rank([rr1, rr2, rr3, rr4])
    expected_mrr = (1.0 + 0.5 + 1/3 + 0.0) / 4.0
    print(f"MRR: {mrr:.4f} (expected {expected_mrr:.4f})")
    assert abs(mrr - expected_mrr) < 1e-5

    print("[SUCCESS] Challenge 104 MRR Passed!")

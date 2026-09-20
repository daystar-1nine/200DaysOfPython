"""
Coding Challenge 104.5: Precision@K.
"""

from typing import Any, List


def precision_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    """Challenge 5: Implement Precision@K.
    
    Formula: |Retrieved_K ∩ Relevant| / K
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return float(hits / k)


if __name__ == "__main__":
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5, 7, 9]

    p1 = precision_at_k(retrieved, relevant, k=1)
    p3 = precision_at_k(retrieved, relevant, k=3)
    p5 = precision_at_k(retrieved, relevant, k=5)

    print(f"P@1: {p1} (expected 1.0)")
    print(f"P@3: {p3} (expected {2/3:.4f})")
    print(f"P@5: {p5} (expected 0.6)")

    assert p1 == 1.0
    assert abs(p3 - 2/3) < 1e-5
    assert p5 == 0.6
    print("[SUCCESS] Challenge 104 Precision@K Passed!")

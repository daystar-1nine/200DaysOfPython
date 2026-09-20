"""
Coding Challenge 104.6: Recall@K.
"""

from typing import Any, List


def recall_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    """Challenge 6: Implement Recall@K.
    
    Formula: |Retrieved_K ∩ Relevant| / |Relevant|
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    rel_set = set(relevant_ids)
    if len(rel_set) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return float(hits / len(rel_set))


if __name__ == "__main__":
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5, 7, 9, 11]  # 6 relevant total

    r1 = recall_at_k(retrieved, relevant, k=1)
    r3 = recall_at_k(retrieved, relevant, k=3)
    r5 = recall_at_k(retrieved, relevant, k=5)

    print(f"R@1: {r1} (expected {1/6:.4f})")
    print(f"R@3: {r3} (expected {2/6:.4f})")
    print(f"R@5: {r5} (expected {3/6:.4f})")

    assert abs(r1 - 1/6) < 1e-5
    assert abs(r3 - 2/6) < 1e-5
    assert abs(r5 - 3/6) < 1e-5
    print("[SUCCESS] Challenge 104 Recall@K Passed!")

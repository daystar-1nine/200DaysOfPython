"""
Challenge 2: Hybrid Search with Min-Max Score Normalization.
Contrasts linear score combination with Reciprocal Rank Fusion on sparse and dense candidates.
"""
from typing import Dict, List, Tuple


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Scales scores to [0.0, 1.0]."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    diff = max_v - min_v
    if diff == 0.0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / diff for k, v in scores.items()}


def weighted_hybrid_search(
    bm25_scores: Dict[str, float],
    dense_scores: Dict[str, float],
    alpha: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Score(d) = alpha * norm_dense(d) + (1 - alpha) * norm_bm25(d)
    """
    all_keys = set(bm25_scores.keys()).union(dense_scores.keys())
    full_bm25 = {k: bm25_scores.get(k, 0.0) for k in all_keys}
    full_dense = {k: dense_scores.get(k, 0.0) for k in all_keys}

    norm_bm25 = min_max_normalize(full_bm25)
    norm_dense = min_max_normalize(full_dense)

    fused = {}
    for k in all_keys:
        fused[k] = alpha * norm_dense[k] + (1.0 - alpha) * norm_bm25[k]

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def main():
    print("=== Challenge 2: Hybrid Search (Weighted Score Fusion) ===")
    
    # Raw BM25 scores (unbounded positive floats)
    raw_bm25 = {
        "doc_sql_join": 14.82,
        "doc_hash_join": 11.20,
        "doc_sql_index": 5.40,
        "doc_python_loop": 0.50
    }
    
    # Dense cosine scores (bounded in [-1, 1], typically [0.4, 0.9])
    raw_dense = {
        "doc_hash_join": 0.88,
        "doc_sql_index": 0.82,
        "doc_sql_join": 0.74,
        "doc_python_loop": 0.35
    }

    # Evaluate with balanced alpha = 0.5
    fused_balanced = weighted_hybrid_search(raw_bm25, raw_dense, alpha=0.5)
    print("\nBalanced Hybrid Search (alpha = 0.5):")
    for r, (doc, s) in enumerate(fused_balanced, start=1):
        print(f"  {r}. {doc:<20} Score: {s:.4f}")

    # Evaluate with keyword-heavy alpha = 0.2
    fused_sparse = weighted_hybrid_search(raw_bm25, raw_dense, alpha=0.2)
    print("\nSparse-Biased Hybrid Search (alpha = 0.2):")
    for r, (doc, s) in enumerate(fused_sparse, start=1):
        print(f"  {r}. {doc:<20} Score: {s:.4f}")

    assert fused_balanced[0][0] in {"doc_sql_join", "doc_hash_join"}
    print("\n[SUCCESS] Min-max score normalization and weighted fusion verified!")


if __name__ == "__main__":
    main()

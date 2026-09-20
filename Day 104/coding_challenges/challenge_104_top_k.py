"""
Coding Challenge 104.3 & 104.4: Cosine Similarity and Top-K Ranking.
"""

from typing import Any, Dict, List
import numpy as np


def cosine_similarity_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Challenge 3: Implement safe cosine similarity.
    
    Returns dot(a, b) / (||a|| * ||b||), returning 0.0 if either norm is zero.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0 or np.isnan(denom):
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_ranking(
    query_vec: np.ndarray,
    doc_vectors: List[np.ndarray],
    documents: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Challenge 4: Implement Top-K ranking.
    
    Computes cosine similarity of query against each document,
    and returns top K results sorted descending by score.
    """
    scores = [cosine_similarity_safe(query_vec, doc_v) for doc_v in doc_vectors]
    ranked_indices = np.argsort(-np.array(scores))[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "rank": rank,
            "document_id": documents[idx].get("id"),
            "title": documents[idx].get("title"),
            "score": float(scores[idx])
        })
    return results


if __name__ == "__main__":
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    v3 = np.array([1.0, 1.0])
    v0 = np.array([0.0, 0.0])

    assert cosine_similarity_safe(v1, v1) == 1.0, "Identical failed"
    assert cosine_similarity_safe(v1, v2) == 0.0, "Orthogonal failed"
    assert cosine_similarity_safe(v1, v0) == 0.0, "Zero norm failed"

    docs = [
        {"id": 1, "title": "Doc 1"},
        {"id": 2, "title": "Doc 2"},
        {"id": 3, "title": "Doc 3"}
    ]
    ranked = top_k_ranking(v1, [v2, v1, v3], docs, top_k=2)
    assert ranked[0]["document_id"] == 2, "Top 1 should be Doc 2"
    assert ranked[1]["document_id"] == 3, "Top 2 should be Doc 3"
    print("Top-K results:", ranked)
    print("[SUCCESS] Challenge 104 Top-K Ranking Passed!")

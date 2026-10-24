"""
Day 117 — Coding Challenges 3, 4, 5, 6: Retrieval & Evaluation Metrics from Scratch
Implements top-k similarity search, Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
"""
from typing import List, Tuple, Sequence
import numpy as np


# -------------------------------------------------------------
# Challenge 3: Top-K Vector Search
# -------------------------------------------------------------
def top_k_similar(
    query_vec: Sequence[float],
    doc_vecs: Sequence[Sequence[float]],
    k: int = 3
) -> List[Tuple[int, float]]:
    """
    Ranks documents by cosine similarity to query_vec and returns top-k indices and scores.

    Returns:
        List of (doc_index, similarity_score) tuples sorted in descending order.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(doc_vecs) == 0:
        return []

    q = np.asarray(query_vec, dtype=np.float32)
    docs = np.asarray(doc_vecs, dtype=np.float32)

    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return [(i, 0.0) for i in range(min(k, len(docs)))]

    docs_norm = np.linalg.norm(docs, axis=1, keepdims=True)
    docs_norm = np.where(docs_norm < 1e-12, 1e-12, docs_norm)

    # Dot product normalized
    sims = np.dot(docs, q) / (docs_norm.squeeze() * q_norm)

    top_indices = np.argsort(sims)[::-1][:k]
    return [(int(idx), float(sims[idx])) for idx in top_indices]


# -------------------------------------------------------------
# Challenge 4: Precision@K
# -------------------------------------------------------------
def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int
) -> float:
    """
    Precision@K = |Retrieved[:k] intersect Relevant| / k
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not retrieved or not relevant:
        return 0.0

    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_k & relevant_set)
    return float(hits / k)


# -------------------------------------------------------------
# Challenge 5: Recall@K
# -------------------------------------------------------------
def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int
) -> float:
    """
    Recall@K = |Retrieved[:k] intersect Relevant| / |Relevant|
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        return 1.0 if not retrieved else 0.0
    if not retrieved:
        return 0.0

    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_k & relevant_set)
    return float(hits / len(relevant_set))


# -------------------------------------------------------------
# Challenge 6: Mean Reciprocal Rank (MRR)
# -------------------------------------------------------------
def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Computes Reciprocal Rank for a single query (1 / first_rank)."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return float(1.0 / rank)
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[Sequence[str]]
) -> float:
    """Computes Mean Reciprocal Rank (MRR) across a query workload."""
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("Lengths of all_retrieved and all_relevant must match.")
    if not all_retrieved:
        return 0.0

    rr_scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    return float(sum(rr_scores) / len(rr_scores))


if __name__ == "__main__":
    print("=== Challenge 3: Top-K Vector Search ===")
    query = [1.0, 0.0, 0.5]
    docs = [
        [0.9, 0.1, 0.4],  # high similarity
        [0.0, 1.0, 0.0],  # orthogonal
        [0.7, 0.0, 0.7],  # high similarity
        [-1.0, 0.0, 0.0]  # negative similarity
    ]
    ranked = top_k_similar(query, docs, k=2)
    for rank, (idx, score) in enumerate(ranked, start=1):
        print(f"  Rank {rank}: Doc {idx} with score {score:.4f}")

    print("\n=== Challenges 4 & 5: Precision@K & Recall@K ===")
    retrieved_docs = ["docA", "docB", "docC", "docD", "docE"]
    relevant_docs = ["docB", "docD", "docF"]

    p3 = precision_at_k(retrieved_docs, relevant_docs, k=3)
    r3 = recall_at_k(retrieved_docs, relevant_docs, k=3)
    p5 = precision_at_k(retrieved_docs, relevant_docs, k=5)
    r5 = recall_at_k(retrieved_docs, relevant_docs, k=5)
    print(f"Retrieved[:3]: {retrieved_docs[:3]} | Relevant: {relevant_docs}")
    print(f"Precision@3: {p3:.4f} | Recall@3: {r3:.4f}")
    print(f"Precision@5: {p5:.4f} | Recall@5: {r5:.4f}")

    print("\n=== Challenge 6: Mean Reciprocal Rank (MRR) ===")
    all_ret = [
        ["docA", "docB", "docC"],  # first hit at rank 2 -> RR = 0.50
        ["docX", "docY", "docZ"],  # first hit at rank 1 (docX) -> RR = 1.00
        ["doc1", "doc2", "doc3"],  # first hit at rank 3 (doc3) -> RR = 0.333
        ["docP", "docQ", "docR"]   # no hit -> RR = 0.00
    ]
    all_rel = [
        ["docB"],
        ["docX"],
        ["doc3"],
        ["docZ"]
    ]
    mrr = mean_reciprocal_rank(all_ret, all_rel)
    print(f"Calculated MRR across 4 queries: {mrr:.4f} (Expected: (0.5 + 1.0 + 0.3333 + 0.0) / 4 = 0.4583)")

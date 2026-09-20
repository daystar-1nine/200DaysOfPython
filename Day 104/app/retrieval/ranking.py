"""
Ranking and similarity computation module for Day 104 Semantic Search Engine.
Provides robust cosine similarity calculation, vectorized scoring, and Top-K ranking.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors with zero-norm guard.
    
    Formula:
        cos(a, b) = (a . b) / (||a|| * ||b||)
        
    Args:
        a: 1D numpy array.
        b: 1D numpy array.
        
    Returns:
        Cosine similarity scalar float between -1.0 and 1.0.
    """
    a_arr = np.asarray(a, dtype=np.float32).flatten()
    b_arr = np.asarray(b, dtype=np.float32).flatten()

    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Shape mismatch: {a_arr.shape} vs {b_arr.shape}")

    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0 or np.isnan(denom):
        return 0.0

    sim = float(np.dot(a_arr, b_arr) / denom)
    return float(np.clip(sim, -1.0, 1.0))


def compute_similarities(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Compute vectorized cosine similarities between a query vector and a matrix of document vectors.
    
    Args:
        query_vec: 1D numpy array of shape (D,).
        doc_matrix: 2D numpy array of shape (N, D).
        
    Returns:
        1D numpy array of shape (N,) containing similarity scores.
    """
    q = np.asarray(query_vec, dtype=np.float32).flatten()
    docs = np.asarray(doc_matrix, dtype=np.float32)

    if docs.ndim != 2:
        raise ValueError(f"doc_matrix must be 2D, got shape {docs.shape}")
    if q.shape[0] != docs.shape[1]:
        raise ValueError(f"Dimension mismatch: query ({q.shape[0]}) vs docs ({docs.shape[1]})")

    n_docs = docs.shape[0]
    if n_docs == 0:
        return np.array([], dtype=np.float32)

    q_norm = np.linalg.norm(q)
    if q_norm == 0.0 or np.isnan(q_norm):
        return np.zeros(n_docs, dtype=np.float32)

    doc_norms = np.linalg.norm(docs, axis=1)
    denominators = q_norm * doc_norms

    # Avoid division by zero
    zero_mask = (denominators == 0.0) | np.isnan(denominators)
    safe_denominators = np.where(zero_mask, 1.0, denominators)

    dot_products = np.dot(docs, q)
    scores = dot_products / safe_denominators
    scores[zero_mask] = 0.0

    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize score array to [0, 1] range using min-max scaling.
    
    Args:
        scores: 1D numpy array of scores.
        
    Returns:
        1D numpy array with values in [0, 1].
    """
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0:
        return s

    min_val = float(np.min(s))
    max_val = float(np.max(s))

    if max_val == min_val:
        return np.zeros_like(s) if min_val <= 0 else np.ones_like(s)

    return (s - min_val) / (max_val - min_val)


def rank_top_k(
    scores: np.ndarray,
    documents: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Rank documents by similarity score and return Top-K results.
    
    Args:
        scores: 1D array of scores for each document.
        documents: List of document dictionaries corresponding to scores.
        top_k: Number of top results to retrieve.
        
    Returns:
        List of dictionaries containing document metadata and ranking information.
    """
    if len(scores) != len(documents):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(documents)} docs")

    if top_k <= 0:
        return []

    # Sort descending by score; use negative score with argsort
    ranked_indices = np.argsort(-scores)
    actual_k = min(top_k, len(documents))

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices[:actual_k], start=1):
        doc = documents[idx]
        snippet = doc.get("text", "")[:120] + ("..." if len(doc.get("text", "")) > 120 else "")
        results.append({
            "rank": rank,
            "document_id": doc.get("id"),
            "title": doc.get("title", ""),
            "category": doc.get("category", ""),
            "score": float(scores[idx]),
            "snippet": snippet
        })

    return results

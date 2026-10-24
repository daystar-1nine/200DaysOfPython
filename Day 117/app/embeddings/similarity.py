"""
Vector similarity functions: Cosine similarity, Euclidean distance, and batch operations.
"""
from typing import Union
import numpy as np


def cosine_similarity(
    vec_a: Union[np.ndarray, list],
    vec_b: Union[np.ndarray, list]
) -> float:
    """
    Computes standard cosine similarity between two 1D vectors:
        sim = (a . b) / (||a|| * ||b||)
    Returns float in range [-1.0, 1.0]. Returns 0.0 if either vector has zero magnitude.
    """
    a = np.asarray(vec_a, dtype=np.float64).flatten()
    b = np.asarray(vec_b, dtype=np.float64).flatten()

    if a.shape != b.shape:
        raise ValueError(f"Dimension mismatch: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = np.dot(a, b)
    sim = dot / (norm_a * norm_b)
    # Clip numerical precision drift
    return float(np.clip(sim, -1.0, 1.0))


# Alias for explicit pairwise calls
cosine_similarity_pair = cosine_similarity


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes pairwise cosine similarity matrix between row vectors of A and B."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    norm_A_safe = np.where(norm_A == 0.0, 1.0, norm_A)
    norm_B_safe = np.where(norm_B == 0.0, 1.0, norm_B)
    dots = np.dot(A, B.T)
    sims = dots / (norm_A_safe * norm_B_safe.T)
    return np.clip(sims, -1.0, 1.0)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes vector to unit L2 norm."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return np.zeros_like(v)
    return v / norm


def batch_cosine_similarity(
    query_vector: Union[np.ndarray, list],
    matrix: np.ndarray
) -> np.ndarray:
    """
    Vectorized cosine similarity between a single 1D query vector and a 2D matrix (N, D).
    Returns a 1D array of length N with similarity scores.
    """
    q = np.asarray(query_vector, dtype=np.float64).flatten()
    M = np.asarray(matrix, dtype=np.float64)

    if M.ndim == 1:
        M = M.reshape(1, -1)

    if M.shape[1] != q.shape[0]:
        raise ValueError(f"Query dim {q.shape[0]} does not match matrix feature dim {M.shape[1]}")

    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        return np.zeros(M.shape[0], dtype=np.float64)

    m_norms = np.linalg.norm(M, axis=1)
    # Safe division avoiding zero-division
    m_norms_safe = np.where(m_norms == 0.0, 1.0, m_norms)

    dots = np.dot(M, q)
    sims = dots / (m_norms_safe * q_norm)
    sims = np.where(m_norms == 0.0, 0.0, sims)

    return np.clip(sims, -1.0, 1.0)


def euclidean_distance(
    vec_a: Union[np.ndarray, list],
    vec_b: Union[np.ndarray, list]
) -> float:
    """Computes Euclidean (L2) distance between two vectors."""
    a = np.asarray(vec_a, dtype=np.float64).flatten()
    b = np.asarray(vec_b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        raise ValueError(f"Dimension mismatch: {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b))


def dot_product_similarity(
    vec_a: Union[np.ndarray, list],
    vec_b: Union[np.ndarray, list]
) -> float:
    """Computes unnormalized inner dot product."""
    a = np.asarray(vec_a, dtype=np.float64).flatten()
    b = np.asarray(vec_b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        raise ValueError(f"Dimension mismatch: {a.shape} vs {b.shape}")
    return float(np.dot(a, b))

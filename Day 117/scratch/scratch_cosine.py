"""
Day 117 — Coding Challenge 2: Pure Cosine Similarity (Zero External Libraries)
Implements cosine similarity between 1D / 2D vectors from first principles using pure math / numpy.
"""
import math
from typing import Sequence, Union
import numpy as np


def cosine_similarity_pure(
    a: Sequence[float],
    b: Sequence[float],
    eps: float = 1e-12
) -> float:
    """
    Computes cosine similarity between two 1D vectors using standard library math.
    cos(theta) = (a . b) / (||a|| * ||b||)
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("Vectors cannot be empty")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a < eps or norm_b < eps:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def cosine_similarity_numpy(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    Computes cosine similarity between two 1D numpy arrays.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < eps or norm_b < eps:
        return 0.0

    return float(dot / (norm_a * norm_b))


if __name__ == "__main__":
    v1 = [1.0, 2.0, 3.0]
    v2 = [2.0, 4.0, 6.0]  # collinear -> 1.0
    v3 = [-1.0, -2.0, -3.0]  # opposite -> -1.0
    v4 = [-3.0, 0.0, 1.0]  # orthogonal -> 0.0

    print("=== Challenge 2: Cosine Similarity from Scratch ===")
    print(f"Collinear v1 & v2 : {cosine_similarity_pure(v1, v2):.6f}")
    print(f"Opposite  v1 & v3 : {cosine_similarity_pure(v1, v3):.6f}")
    print(f"Orthogonal v1 & v4: {cosine_similarity_pure(v1, v4):.6f}")
    print(f"NumPy Verification: {cosine_similarity_numpy(np.array(v1), np.array(v2)):.6f}")

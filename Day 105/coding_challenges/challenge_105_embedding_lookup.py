"""
Coding Challenge 105.6: Embedding Lookup using NumPy.
Day 105: Neural NLP & Text Classification.
"""

from typing import List, Union
import numpy as np


def challenge_embedding_lookup(
    embedding_matrix: np.ndarray,
    token_ids: Union[List[int], np.ndarray]
) -> np.ndarray:
    """Challenge 6: Implement embedding lookup using NumPy.
    
    Given:
        embedding_matrix: Shape (V, D)
        token_ids: Shape (T,) or (B, T)
    Returns:
        Dense array of shape (T, D) or (B, T, D)
    """
    matrix = np.asarray(embedding_matrix, dtype=np.float32)
    ids = np.asarray(token_ids, dtype=np.int64)
    return matrix[ids]


if __name__ == "__main__":
    np.random.seed(42)
    W = np.random.randn(10, 4).astype(np.float32)
    ids = [2, 5, 7]

    looked_up = challenge_embedding_lookup(W, ids)
    print("W shape:", W.shape)
    print("Lookup IDs:", ids)
    print("Looked up shape:", looked_up.shape)

    assert looked_up.shape == (3, 4)
    assert np.allclose(looked_up[0], W[2])
    assert np.allclose(looked_up[1], W[5])
    assert np.allclose(looked_up[2], W[7])

    print("[SUCCESS] Challenge 105.6 Passed!")

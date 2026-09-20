"""
Coding Challenge 104.1 & 104.2: Document Embeddings (Mean & Weighted Pooling).
"""

from typing import Dict, List, Optional
import numpy as np


def document_embedding_mean(
    tokens: List[str],
    embeddings: Dict[str, np.ndarray],
    dim: int = 3
) -> np.ndarray:
    """Challenge 1: Implement mean document embeddings.
    
    Given a list of tokens and a dictionary of word embeddings,
    return the arithmetic mean vector. If no tokens exist in embeddings,
    return a zero vector of dimension `dim`.
    """
    valid_vectors = [embeddings[t] for t in tokens if t in embeddings]
    if not valid_vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(valid_vectors, axis=0), axis=0).astype(np.float32)


def document_embedding_weighted(
    tokens: List[str],
    embeddings: Dict[str, np.ndarray],
    weights: Dict[str, float],
    dim: int = 3
) -> np.ndarray:
    """Challenge 2: Implement weighted document embeddings.
    
    Weights each word embedding vector by its scalar weight (e.g. TF-IDF),
    sums them, and normalizes by the sum of weights.
    """
    valid_vectors = []
    valid_weights = []
    for t in tokens:
        if t in embeddings:
            valid_vectors.append(embeddings[t])
            valid_weights.append(float(weights.get(t, 1.0)))

    if not valid_vectors or sum(valid_weights) == 0.0:
        return np.zeros(dim, dtype=np.float32)

    v_arr = np.stack(valid_vectors, axis=0)
    w_arr = np.array(valid_weights, dtype=np.float32)[:, np.newaxis]
    return (np.sum(v_arr * w_arr, axis=0) / np.sum(w_arr)).astype(np.float32)


if __name__ == "__main__":
    embeddings = {
        "cat": np.array([0.2, 0.4, 0.1]),
        "dog": np.array([0.3, 0.5, 0.2])
    }
    tokens = ["cat", "dog"]
    mean_vec = document_embedding_mean(tokens, embeddings, dim=3)
    print("Mean embedding:", mean_vec)
    assert np.allclose(mean_vec, [0.25, 0.45, 0.15]), "Mean pooling failed!"

    weights = {"cat": 2.0, "dog": 1.0}
    weighted_vec = document_embedding_weighted(tokens, embeddings, weights, dim=3)
    print("Weighted embedding:", weighted_vec)
    # Expected: (2 * [0.2, 0.4, 0.1] + 1 * [0.3, 0.5, 0.2]) / 3 = [0.7/3, 1.3/3, 0.4/3]
    expected_weighted = np.array([0.7 / 3, 1.3 / 3, 0.4 / 3])
    assert np.allclose(weighted_vec, expected_weighted), "Weighted pooling failed!"
    print("[SUCCESS] Challenge 104 Document Embedding Passed!")

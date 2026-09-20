"""
Pooling module for Day 104 Semantic Search Engine.
Implements mean pooling, weighted pooling, and explicit <UNK> token strategies.
"""

from typing import Dict, List, Optional, Union
import numpy as np


def mean_pooling(
    tokens: List[str],
    embeddings: Dict[str, np.ndarray],
    embedding_dim: int = 64,
    unk_vector: Optional[np.ndarray] = None,
    unk_strategy: str = "zero"
) -> np.ndarray:
    """Compute document embedding via mean pooling over token embeddings.
    
    Formula:
        d = (1 / |T|) * sum_{w in T} v_w
        
    Args:
        tokens: List of word tokens.
        embeddings: Mapping from word token to 1D numpy array.
        embedding_dim: Dimensionality of embeddings.
        unk_vector: Optional predefined vector for unknown tokens.
        unk_strategy: Strategy for handling out-of-vocabulary words ('zero', 'ignore', 'unk').
        
    Returns:
        1D numpy array of shape (embedding_dim,).
    """
    if not tokens:
        return np.zeros(embedding_dim, dtype=np.float32)

    vectors: List[np.ndarray] = []

    for token in tokens:
        if token in embeddings:
            vectors.append(embeddings[token])
        else:
            if unk_strategy == "unk" and unk_vector is not None:
                vectors.append(unk_vector)
            # For 'zero' or 'ignore', we skip out-of-vocabulary tokens

    if not vectors:
        # If no tokens were recognized in the vocabulary
        if unk_strategy == "unk" and unk_vector is not None:
            return np.array(unk_vector, dtype=np.float32)
        return np.zeros(embedding_dim, dtype=np.float32)

    stacked = np.stack(vectors, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def weighted_pooling(
    tokens: List[str],
    embeddings: Dict[str, np.ndarray],
    weights: Union[Dict[str, float], List[float]],
    embedding_dim: int = 64,
    unk_vector: Optional[np.ndarray] = None,
    unk_strategy: str = "zero"
) -> np.ndarray:
    """Compute document embedding via weighted pooling (e.g. TF-IDF weighted).
    
    Formula:
        d = sum_{w in T} (weight_w * v_w) / sum_{w in T} weight_w
        
    Args:
        tokens: List of word tokens.
        embeddings: Mapping from word token to 1D numpy array.
        weights: Dictionary mapping token to weight, or list of weights matching tokens.
        embedding_dim: Dimensionality of embeddings.
        unk_vector: Optional predefined vector for unknown tokens.
        unk_strategy: Strategy for handling out-of-vocabulary words ('zero', 'ignore', 'unk').
        
    Returns:
        1D numpy array of shape (embedding_dim,).
    """
    if not tokens:
        return np.zeros(embedding_dim, dtype=np.float32)

    vectors: List[np.ndarray] = []
    token_weights: List[float] = []

    for idx, token in enumerate(tokens):
        if isinstance(weights, dict):
            weight = float(weights.get(token, 1.0))
        elif isinstance(weights, list) and idx < len(weights):
            weight = float(weights[idx])
        else:
            weight = 1.0

        if token in embeddings:
            vectors.append(embeddings[token])
            token_weights.append(weight)
        else:
            if unk_strategy == "unk" and unk_vector is not None:
                vectors.append(unk_vector)
                token_weights.append(weight)

    if not vectors or sum(token_weights) == 0.0:
        return np.zeros(embedding_dim, dtype=np.float32)

    w_arr = np.array(token_weights, dtype=np.float32)[:, np.newaxis]
    v_arr = np.stack(vectors, axis=0)

    weighted_sum = np.sum(v_arr * w_arr, axis=0)
    total_weight = np.sum(w_arr)

    return (weighted_sum / total_weight).astype(np.float32)

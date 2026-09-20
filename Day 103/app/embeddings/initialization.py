"""Embedding weight initialization."""
import numpy as np
from typing import Tuple

def initialize_embeddings(
    vocab_size: int,
    embedding_dim: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes input embeddings W_in and output embeddings W_out.
    Uses Xavier/uniform initialization scaled by 1 / sqrt(embedding_dim).
    """
    rng = np.random.default_rng(seed)
    limit = 1.0 / np.sqrt(embedding_dim)
    W_in = rng.uniform(-limit, limit, (vocab_size, embedding_dim)).astype(np.float64)
    W_out = rng.uniform(-limit, limit, (vocab_size, embedding_dim)).astype(np.float64)
    return W_in, W_out

"""
Embedding lookup module for Day 105: Neural NLP & Text Classification.
Provides vectorized index selection for token embedding matrices.
"""

from typing import List, Union
import numpy as np


def embedding_lookup(
    embedding_matrix: np.ndarray,
    token_ids: Union[List[int], np.ndarray]
) -> np.ndarray:
    """Retrieve dense embedding vectors for integer token IDs using NumPy indexing.
    
    Args:
        embedding_matrix: 2D array of shape (vocab_size, embedding_dim).
        token_ids: 1D array of shape (seq_len,) or 2D array of shape (batch_size, seq_len).
        
    Returns:
        Dense array of shape (seq_len, embedding_dim) or (batch_size, seq_len, embedding_dim).
        
    Raises:
        ValueError: If embedding_matrix is not 2D.
        IndexError: If any token ID is out of vocabulary bounds.
    """
    matrix = np.asarray(embedding_matrix, dtype=np.float32)
    ids = np.asarray(token_ids, dtype=np.int64)

    if matrix.ndim != 2:
        raise ValueError(f"embedding_matrix must be 2D, got shape {matrix.shape}")

    vocab_size, emb_dim = matrix.shape

    if ids.size > 0:
        if np.any(ids < 0) or np.any(ids >= vocab_size):
            min_id = int(np.min(ids))
            max_id = int(np.max(ids))
            raise IndexError(f"Token ID out of bounds [0, {vocab_size - 1}]. Got min: {min_id}, max: {max_id}")

    # Advanced indexing retrieves rows preserving batch/sequence shape
    return matrix[ids]

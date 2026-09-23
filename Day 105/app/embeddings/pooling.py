"""
Masked pooling module for Day 105: Neural NLP & Text Classification.
Computes masked global average pooling so padding tokens do not distort representations.
"""

from typing import Union
import numpy as np


def masked_global_average_pooling_np(
    embeddings: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Compute masked global average pooling across sequence dimension using NumPy.
    
    Formula:
        h = sum_{t=1}^T (mask_t * e_t) / sum_{t=1}^T mask_t
        
    Args:
        embeddings: Array of shape (seq_len, dim) or (batch_size, seq_len, dim).
        mask: Array of shape (seq_len,) or (batch_size, seq_len) with 1s and 0s.
        
    Returns:
        Pooled array of shape (dim,) or (batch_size, dim).
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)

    if emb.ndim == 2 and m.ndim == 1:
        # Single sequence: (T, D) and (T,)
        if emb.shape[0] != m.shape[0]:
            raise ValueError(f"Sequence length mismatch: {emb.shape[0]} vs {m.shape[0]}")
        m_expanded = m[:, np.newaxis]  # (T, 1)
        weighted_sum = np.sum(emb * m_expanded, axis=0)  # (D,)
        total_tokens = np.sum(m)
        denom = max(total_tokens, 1e-9)
        return (weighted_sum / denom).astype(np.float32)

    elif emb.ndim == 3 and m.ndim == 2:
        # Batch: (B, T, D) and (B, T)
        if emb.shape[:2] != m.shape:
            raise ValueError(f"Shape mismatch: {emb.shape[:2]} vs {m.shape}")
        m_expanded = m[:, :, np.newaxis]  # (B, T, 1)
        weighted_sum = np.sum(emb * m_expanded, axis=1)  # (B, D)
        total_tokens = np.sum(m, axis=1, keepdims=True)  # (B, 1)
        safe_denom = np.where(total_tokens == 0, 1.0, total_tokens)
        pooled = np.where(total_tokens == 0, 0.0, weighted_sum / safe_denom)
        return pooled.astype(np.float32)

    else:
        raise ValueError(f"Unsupported dimensions: embeddings {emb.shape}, mask {m.shape}")


def masked_global_average_pooling(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    mask: Union[np.ndarray, "torch.Tensor"]
):
    """Universal dispatcher for masked global average pooling."""
    try:
        import torch
        if isinstance(embeddings, torch.Tensor) and isinstance(mask, torch.Tensor):
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
            weighted_sum = torch.sum(embeddings * mask_expanded, dim=1)  # (B, D)
            token_counts = torch.sum(mask.float(), dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            return weighted_sum / token_counts
    except ImportError:
        pass

    return masked_global_average_pooling_np(embeddings, mask)

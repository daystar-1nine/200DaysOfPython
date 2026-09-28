"""
Padding, truncation, and masking utilities for sequence data.
"""
from typing import List, Tuple
import numpy as np
import torch


def pad_sequences(
    sequences: List[List[int]],
    max_length: int = 100,
    pad_id: int = 0,
    truncating: str = "post",
    padding: str = "post"
) -> np.ndarray:
    """
    Pads or truncates lists of integer sequences to uniform length.
    """
    batch_size = len(sequences)
    padded = np.full((batch_size, max_length), pad_id, dtype=np.int64)

    for i, seq in enumerate(sequences):
        if not seq:
            continue
        # Truncation
        if len(seq) > max_length:
            if truncating == "post":
                truncated = seq[:max_length]
            else:
                truncated = seq[-max_length:]
        else:
            truncated = seq

        # Padding
        if padding == "post":
            padded[i, :len(truncated)] = truncated
        else:
            padded[i, max_length - len(truncated):] = truncated

    return padded


def create_padding_mask(token_ids: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """
    Creates a boolean mask where True indicates padding positions.
    Shape: (batch_size, sequence_length)
    """
    return np.asarray(token_ids) == pad_id


def create_padding_mask_torch(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    PyTorch version of create_padding_mask.
    Returns boolean tensor of shape (batch_size, sequence_length), True at pad locations.
    """
    return token_ids == pad_id


def causal_mask(length: int) -> np.ndarray:
    """
    Creates an upper triangular boolean mask for causal autoregression.
    Shape: (length, length), where True represents positions that should be masked.
    """
    return np.triu(np.ones((length, length), dtype=bool), k=1)

"""
Sequence padding and truncation module for Day 105: Neural NLP & Text Classification.
Ensures uniform sequence tensor shapes across mini-batches while creating boolean masks.
"""

from typing import List, Tuple, Union
import numpy as np


def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_id: int = 0,
    padding_strategy: str = "post",
    truncation_strategy: str = "post"
) -> Tuple[List[int], List[int]]:
    """Pad or truncate a single integer sequence to a fixed maximum length.
    
    Args:
        sequence: List of integer token IDs.
        max_length: Target sequence length.
        pad_id: Special padding integer ID (default 0).
        padding_strategy: 'post' (pad at end) or 'pre' (pad at start).
        truncation_strategy: 'post' (truncate from end) or 'pre' (truncate from start).
        
    Returns:
        Tuple of (padded_sequence, mask) where mask is 1 for real tokens, 0 for padding.
    """
    if max_length <= 0:
        return [], []

    seq = list(sequence)
    seq_len = len(seq)

    # 1. Truncate if necessary
    if seq_len > max_length:
        if truncation_strategy == "pre":
            seq = seq[-max_length:]
        else:
            seq = seq[:max_length]
        mask = [1] * max_length
        return seq, mask

    # 2. Pad if necessary
    pad_len = max_length - seq_len
    if pad_len > 0:
        if padding_strategy == "pre":
            padded = [pad_id] * pad_len + seq
            mask = [0] * pad_len + [1] * seq_len
        else:
            padded = seq + [pad_id] * pad_len
            mask = [1] * seq_len + [0] * pad_len
        return padded, mask

    # 3. Exact match
    return seq, [1] * max_length


def create_mask(padded_sequence: List[int], pad_id: int = 0) -> List[int]:
    """Generate a binary mask where real tokens are 1 and padding tokens are 0.
    
    Args:
        padded_sequence: Padded integer list.
        pad_id: Padding token identifier.
        
    Returns:
        Binary list of 1s and 0s.
    """
    return [0 if x == pad_id else 1 for x in padded_sequence]


def pad_batch(
    sequences: List[List[int]],
    max_length: int,
    pad_id: int = 0,
    padding_strategy: str = "post",
    truncation_strategy: str = "post"
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad and mask an entire batch of sequences into NumPy arrays.
    
    Args:
        sequences: List of variable-length integer sequences.
        max_length: Desired sequence length.
        pad_id: Padding ID.
        padding_strategy: 'post' or 'pre'.
        truncation_strategy: 'post' or 'pre'.
        
    Returns:
        Tuple of (padded_batch_array, masks_batch_array) both with shape (batch_size, max_length).
    """
    batch_padded = []
    batch_masks = []

    for seq in sequences:
        p, m = pad_sequence(
            seq,
            max_length=max_length,
            pad_id=pad_id,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy
        )
        batch_padded.append(p)
        batch_masks.append(m)

    return (
        np.array(batch_padded, dtype=np.int64),
        np.array(batch_masks, dtype=np.float32)
    )

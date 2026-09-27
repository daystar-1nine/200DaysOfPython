import numpy as np
from typing import List
from app.config import PAD_IDX

def pad_sequence(
    seq: List[int],
    max_len: int,
    pad_value: int = PAD_IDX,
    padding: str = "post",
    truncating: str = "post"
) -> List[int]:
    """
    Pads or truncates a single sequence of integer token indices.
    """
    # Truncate
    if len(seq) > max_len:
        if truncating == "post":
            seq = seq[:max_len]
        else:
            seq = seq[-max_len:]
            
    # Pad
    if len(seq) < max_len:
        pad_len = max_len - len(seq)
        if padding == "post":
            seq = seq + [pad_value] * pad_len
        else:
            seq = [pad_value] * pad_len + seq
            
    return seq

def pad_sequences(
    sequences: List[List[int]],
    max_len: int,
    pad_value: int = PAD_IDX,
    padding: str = "post",
    truncating: str = "post"
) -> np.ndarray:
    """
    Pads or truncates a batch of token sequences into a 2D numpy array of shape (N, max_len).
    """
    padded = [
        pad_sequence(s, max_len, pad_value=pad_value, padding=padding, truncating=truncating)
        for s in sequences
    ]
    return np.array(padded, dtype=np.int64)

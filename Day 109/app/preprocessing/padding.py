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
    if len(seq) > max_len:
        seq = seq[:max_len] if truncating == "post" else seq[-max_len:]
    if len(seq) < max_len:
        pad_len = max_len - len(seq)
        seq = seq + [pad_value] * pad_len if padding == "post" else [pad_value] * pad_len + seq
    return seq

def pad_sequences(
    sequences: List[List[int]],
    max_len: int,
    pad_value: int = PAD_IDX,
    padding: str = "post",
    truncating: str = "post"
) -> np.ndarray:
    padded = [
        pad_sequence(s, max_len, pad_value=pad_value, padding=padding, truncating=truncating)
        for s in sequences
    ]
    return np.array(padded, dtype=np.int64)

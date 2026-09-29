"""
Causal Mask implementations in pure NumPy and PyTorch.
Ensures that token at position i can only attend to positions j <= i.
"""
from typing import Tuple
import numpy as np
import torch


def causal_mask_numpy(sequence_length: int) -> np.ndarray:
    """
    Constructs a lower-triangular binary causal mask of shape (seq_len, seq_len) in NumPy.
    1 indicates allowed attention (past and current positions).
    0 indicates blocked attention (future positions).
    """
    return np.tril(np.ones((sequence_length, sequence_length), dtype=np.int32))


def causal_mask_torch(sequence_length: int, device: torch.device = None) -> torch.Tensor:
    """
    Constructs a lower-triangular binary causal mask of shape (seq_len, seq_len) in PyTorch.
    """
    return torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))


def create_additive_causal_mask(sequence_length: int, device: torch.device = None) -> torch.Tensor:
    """
    Constructs an additive attention mask of shape (seq_len, seq_len) with:
      0.0 for allowed positions (j <= i)
      -inf for forbidden future positions (j > i)
    """
    mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = torch.nan_to_num(mask, nan=0.0) # upper triangle is -inf, lower is 0.0
    # Equivalently:
    tril = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))
    additive_mask = torch.zeros(sequence_length, sequence_length, device=device)
    additive_mask = additive_mask.masked_fill(~tril, float("-inf"))
    return additive_mask


# Function alias requested by curriculum
causal_mask = causal_mask_numpy


if __name__ == "__main__":
    n = 4
    mask_np = causal_mask(n)
    print("NumPy Causal Mask (4x4):")
    print(mask_np)
    assert mask_np.shape == (4, 4)
    assert np.all(np.diag(mask_np) == 1)
    assert mask_np[0, 1] == 0
    assert mask_np[3, 0] == 1

    mask_pt = create_additive_causal_mask(n)
    print("\nPyTorch Additive Causal Mask (4x4):")
    print(mask_pt)
    assert mask_pt[0, 0].item() == 0.0
    assert mask_pt[0, 1].item() == float("-inf")
    print("\nCausal mask implementations verified!")

"""
Day 112 - Coding Challenge 1: Causal Mask Construction
Problem: Construct lower-triangular causal masks in NumPy and PyTorch with 0 / -inf representation.
"""
import numpy as np
import torch
import torch.nn.functional as F


def causal_mask_numpy(seq_len: int) -> np.ndarray:
    """Returns a (seq_len, seq_len) lower-triangular binary mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=np.int32))


def causal_mask_torch_additive(seq_len: int) -> torch.Tensor:
    """Returns an additive mask with 0.0 on lower triangle and -inf on upper triangle."""
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)), float("-inf"))
    return mask


if __name__ == "__main__":
    T = 4
    np_mask = causal_mask_numpy(T)
    assert np_mask.shape == (T, T)
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert np_mask[i, j] == 1
            else:
                assert np_mask[i, j] == 0

    torch_mask = causal_mask_torch_additive(T)
    assert torch_mask.shape == (T, T)
    assert torch_mask[0, 0] == 0.0
    assert torch.isneginf(torch_mask[0, 1])

    # Softmax test
    scores = torch.randn(1, 1, T, T)
    weights = F.softmax(scores + torch_mask, dim=-1)
    assert torch.all(torch.triu(weights[0, 0], diagonal=1) == 0.0)
    print("Challenge 1: PASSED")

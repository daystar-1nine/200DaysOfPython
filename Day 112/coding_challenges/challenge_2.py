"""
Day 112 - Coding Challenge 2: Shifted Target Alignment for Teacher Forcing
Problem: Given 1D sequence tensor, slice into mini-batches of inputs x and targets y shifted by 1.
"""
from typing import Tuple
import torch


def get_causal_lm_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly samples sequence chunks where:
    x = data[i : i + context_length]
    y = data[i + 1 : i + 1 + context_length]
    """
    torch.manual_seed(seed)
    max_idx = len(data) - context_length - 1
    ix = torch.randint(0, max_idx + 1, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])
    return x, y


if __name__ == "__main__":
    data = torch.arange(100)
    x, y = get_causal_lm_batch(data, batch_size=4, context_length=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # Target at step t must equal input at step t + 1
    for b in range(4):
        for t in range(7):
            assert y[b, t].item() == x[b, t + 1].item()
    print("Challenge 2: PASSED")

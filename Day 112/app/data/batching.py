"""
Batch sampling and dataset iteration with shifted inputs and targets.
Input: x = tokens[:, :-1]
Target: y = tokens[:, 1:]
"""
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly samples mini-batches of sequence chunks with shifted targets.

    Args:
        data: 1D tensor of token IDs
        batch_size: Number of sequences per batch
        context_length: Sequence window size (T)
        device: Target execution device
        seed: Optional RNG seed
    Returns:
        x: Tensor of shape (batch_size, context_length)
        y: Shifted target tensor of shape (batch_size, context_length)
    """
    if len(data) <= context_length:
        raise ValueError(f"Data length ({len(data)}) must be strictly greater than context_length ({context_length})")

    if seed is not None:
        torch.manual_seed(seed)

    max_idx = len(data) - context_length
    ix = torch.randint(0, max_idx, (batch_size,))

    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    return x, y


class CausalLMDataset(Dataset):
    """
    Sequential window dataset for PyTorch DataLoader integration.
    """
    def __init__(self, data: torch.Tensor, context_length: int, stride: Optional[int] = None):
        self.data = data
        self.context_length = context_length
        self.stride = stride or context_length
        self.num_samples = max(0, (len(data) - context_length) // self.stride)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start:start + self.context_length]
        y = self.data[start + 1:start + self.context_length + 1]
        return x, y

"""
Standalone sampling strategies from scratch in NumPy and PyTorch:
- Greedy argmax
- Temperature scaling
- Top-k sampling
- Top-p (Nucleus) sampling
"""
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


def sample_temperature_np(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Samples next token ID using temperature scaling in pure NumPy."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs))


def sample_top_k_np(logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
    """Samples next token ID restricted to top-k logits in pure NumPy."""
    if k <= 0 or k >= len(logits):
        return sample_temperature_np(logits, temperature)

    indices = np.argsort(logits)[-k:]
    top_logits = logits[indices]
    if temperature > 0:
        top_logits = top_logits / temperature
    exp_logits = np.exp(top_logits - np.max(top_logits))
    probs = exp_logits / np.sum(exp_logits)
    chosen_idx = np.random.choice(len(indices), p=probs)
    return int(indices[chosen_idx])


def sample_top_p_np(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
    """Samples next token ID restricted to nucleus top-p probability mass in pure NumPy."""
    if p >= 1.0:
        return sample_temperature_np(logits, temperature)

    scaled_logits = logits if temperature <= 0 else logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    # Keep at least one token, and all tokens whose cumulative probability is within p
    cutoff_idx = np.searchsorted(cumulative_probs, p)
    kept_indices = sorted_indices[:cutoff_idx + 1]
    kept_probs = sorted_probs[:cutoff_idx + 1]
    kept_probs = kept_probs / np.sum(kept_probs)

    chosen = np.random.choice(len(kept_indices), p=kept_probs)
    return int(kept_indices[chosen])


def sample_logits_torch(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Applies temperature scaling, top-k filtering, and top-p (nucleus) filtering to logits,
    then samples token IDs from the resulting categorical distribution.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        temperature: Temperature parameter (> 0). If 0 or None, defaults to greedy argmax.
        top_k: Optional number of top tokens to retain.
        top_p: Optional cumulative probability threshold for nucleus sampling.
    Returns:
        Tensor of shape (batch_size, 1) containing selected token IDs.
    """
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / max(temperature, 1e-5)

    # Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1, -1.0, 5.0])
    print("NumPy Temperature sample:", sample_temperature_np(logits, 1.0))
    print("NumPy Top-k sample:      ", sample_top_k_np(logits, k=2))
    print("NumPy Top-p sample:      ", sample_top_p_np(logits, p=0.8))

    t_logits = torch.tensor([[2.0, 1.0, 0.1, -1.0, 5.0]])
    sample_pt = sample_logits_torch(t_logits, temperature=0.7, top_k=3, top_p=0.9)
    print("PyTorch Sampled token:   ", sample_pt.item())
    assert 0 <= sample_pt.item() < 5
    print("Sampling strategies scratch verified successfully!")

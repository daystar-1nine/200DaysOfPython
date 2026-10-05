"""
Masked loss and evaluation metrics for Supervised Fine-Tuning.
Computes cross-entropy loss, token-level accuracy, and perplexity exclusively
over active (assistant) tokens.
"""
import math
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F


def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> Tuple[torch.Tensor, float, int]:
    """
    Computes cross-entropy loss and perplexity over non-ignored target tokens:
      logits: [B, T, V]
      targets: [B, T] (where prompt and pad tokens are set to ignore_index=-100)
    Returns:
      (loss_tensor, perplexity_float, active_token_count)
    """
    # Shift so that tokens < t predict t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()

    # Flatten tensors
    vocab_size = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_targets = shift_targets.view(-1)

    # Active token mask
    active_mask = flat_targets != ignore_index
    num_active_tokens = int(active_mask.sum().item())

    if num_active_tokens == 0:
        # Fallback if no target tokens in batch
        dummy_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return dummy_loss, 1.0, 0

    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)

    # Compute perplexity
    clamped_loss = min(float(loss.item()), 20.0)
    perplexity = round(math.exp(clamped_loss), 4)

    return loss, perplexity, num_active_tokens


def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Computes top-1 next-token prediction accuracy on assistant tokens."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()

    preds = torch.argmax(shift_logits, dim=-1)
    mask = shift_targets != ignore_index

    total = int(mask.sum().item())
    if total == 0:
        return 0.0

    correct = int(((preds == shift_targets) & mask).sum().item())
    return round(float(correct / total), 4)

"""
Masking utilities for BERT Masked Language Modeling and attention processing.
"""
from typing import Tuple, Optional, List
import numpy as np
import torch


def apply_bert_mlm_mask(
    input_ids: torch.Tensor,
    mask_token_id: int = 103,
    vocab_size: int = 30522,
    mask_prob: float = 0.15,
    special_tokens: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of the BERT 80/10/10 MLM masking scheme.
    Args:
        input_ids: Tensor of shape (batch_size, seq_len)
        mask_token_id: ID for [MASK] (103)
        vocab_size: Total vocabulary size
        mask_prob: Probability of selecting a token for prediction (0.15)
        special_tokens: List of special token IDs to exclude from masking
    Returns:
        masked_input_ids: Tensor with tokens masked
        labels: Tensor where non-selected tokens are set to -100
    """
    if special_tokens is None:
        special_tokens = [0, 101, 102, 103]

    inputs = input_ids.clone()
    labels = torch.full_like(inputs, fill_value=-100)

    # Eligible candidate tokens
    candidate_mask = torch.ones_like(inputs, dtype=torch.bool)
    for st in special_tokens:
        candidate_mask = candidate_mask & (inputs != st)

    # Random probability tensor
    rand_probs = torch.rand_like(inputs, dtype=torch.float32)
    selected_mask = candidate_mask & (rand_probs < mask_prob)

    # If no tokens were selected in a sequence but candidates exist, pick at least one
    for i in range(inputs.size(0)):
        if not selected_mask[i].any() and candidate_mask[i].any():
            cand_indices = torch.where(candidate_mask[i])[0]
            chosen_idx = cand_indices[torch.randint(0, len(cand_indices), (1,)).item()]
            selected_mask[i, chosen_idx] = True

    # Assign labels to selected positions
    labels[selected_mask] = inputs[selected_mask]

    # 80 / 10 / 10 strategy
    decision_rand = torch.rand_like(inputs, dtype=torch.float32)

    # 80%: replace with [MASK]
    mask_80 = selected_mask & (decision_rand < 0.8)
    inputs[mask_80] = mask_token_id

    # 10%: replace with random token (0.8 <= p < 0.9)
    random_10 = selected_mask & (decision_rand >= 0.8) & (decision_rand < 0.9)
    random_tokens = torch.randint(low=104, high=vocab_size, size=inputs.shape, dtype=inputs.dtype)
    inputs[random_10] = random_tokens[random_10]

    # 10%: keep unchanged (decision_rand >= 0.9)

    return inputs, labels

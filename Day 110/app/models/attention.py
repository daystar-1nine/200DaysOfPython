"""
PyTorch Scaled Dot-Product Attention implementation.
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention in PyTorch:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len_q, d_k)
        k: Key tensor of shape (batch, num_heads, seq_len_k, d_k)
        v: Value tensor of shape (batch, num_heads, seq_len_v, d_k)
        mask: Optional mask (True/1 at positions to mask, False/0 to keep)
        dropout_p: Dropout probability applied to attention weights
        training: Whether model is in training mode
    Returns:
        context: (batch, num_heads, seq_len_q, d_k)
        weights: (batch, num_heads, seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dim() == 2:
            # (batch, seq_len_k) -> (batch, 1, 1, seq_len_k)
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # In PyTorch, masked_fill with float('-inf') on boolean mask
        scores = scores.masked_fill(mask.bool(), float("-1e9"))

    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        weights_dropped = F.dropout(weights, p=dropout_p)
    else:
        weights_dropped = weights

    context = torch.matmul(weights_dropped, v)
    return context, weights

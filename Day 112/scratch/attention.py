"""
Causal Multi-Head Self-Attention implementation from scratch.
Implements scaled dot-product attention with lower-triangular autoregressive causal masking.
"""
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_causal_attention_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy implementation of scaled dot-product causal attention.
    q, k, v: shape (..., seq_len, head_dim)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)

    # Numerically stable softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = np.matmul(weights, v)
    return output, weights


class CausalSelfAttentionScratch(nn.Module):
    """
    PyTorch multi-head causal self-attention module.
    Projects input to Q, K, V, applies lower-triangular causal masking,
    computes attention weights, and projects output.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        context_length: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower triangular matrix (1, 1, context_length, context_length)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
            .view(1, 1, context_length, context_length)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            return_weights: Whether to return the attention probability matrix
        Returns:
            y: Output tensor of shape (batch_size, seq_len, embed_dim)
            att_weights: Optional attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        B, T, C = x.size()

        # Compute Query, Key, Value for all heads
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask: mask positions where bias == 0 to -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over key dimension
        att = F.softmax(att, dim=-1)
        att_weights = att if return_weights else None
        att = self.attn_dropout(att)

        # Attention weighted sum of values: (B, H, T, head_dim)
        y = att @ v

        # Concatenate heads and project: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, att_weights


CausalSelfAttention = CausalSelfAttentionScratch


if __name__ == "__main__":
    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    attn = CausalSelfAttentionScratch(embed_dim=C, num_heads=4, context_length=16)
    out, weights = attn(x, return_weights=True)

    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
    print("Attention weights shape:", weights.shape)
    assert out.shape == (B, T, C)
    assert weights.shape == (B, 4, T, T)

    # Verify that future attention weights are strictly zero
    upper_triangle = torch.triu(weights[0, 0], diagonal=1)
    assert torch.all(upper_triangle == 0.0), "Future positions must receive zero attention!"
    print("CausalSelfAttention verified successfully!")

"""
Production-grade Multi-Head Causal Self-Attention module for MiniGPT.
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with lower-triangular autoregressive masking.
    Ensures that token at position i cannot attend to any future token j > i.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        context_length: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer
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
            x: Tensor of shape (batch_size, seq_len, embed_dim)
            return_weights: If True, returns the attention matrix (B, H, T, T)
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Separate heads: (B, H, T, d_k)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply lower-triangular causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        att_weights = att if return_weights else None
        att = self.attn_dropout(att)

        # Values aggregation
        y = att @ v

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, att_weights

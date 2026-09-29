"""
Day 112 - Coding Challenge 3: Multi-Head Causal Self-Attention
Problem: Implement forward pass of multi-head causal self-attention with QKV projection split.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniCausalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_length: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


if __name__ == "__main__":
    attn = MiniCausalAttention(embed_dim=32, num_heads=4, context_length=16)
    x = torch.randn(2, 6, 32)
    y = attn(x)
    assert y.shape == (2, 6, 32)
    assert not torch.isnan(y).any()
    print("Challenge 3: PASSED")

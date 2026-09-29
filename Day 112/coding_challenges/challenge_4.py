"""
Day 112 - Coding Challenge 4: Pre-LN Transformer Block
Problem: Implement Pre-LN Transformer block with residual stream addition:
         x = x + Attn(LN1(x))
         x = x + MLP(LN2(x))
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from challenge_3 import MiniCausalAttention


class GELUMLP(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class PreLNTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_length: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MiniCausalAttention(embed_dim, num_heads, context_length)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = GELUMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual branch 1
        x = x + self.attn(self.ln_1(x))
        # Pre-LN residual branch 2
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == "__main__":
    block = PreLNTransformerBlock(embed_dim=32, num_heads=4, context_length=16)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()
    print("Challenge 4: PASSED")

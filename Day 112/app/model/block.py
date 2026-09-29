"""
Production Pre-LayerNorm Transformer Block and Feed-Forward Network.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from app.model.attention import CausalSelfAttention


class MLP(nn.Module):
    """
    Two-layer feed-forward network with 4x intermediate expansion and GELU activation.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Decoder Block with Causal Multi-Head Self-Attention and MLP.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        context_length: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            return_weights: Whether to return self-attention weights
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        normed_x = self.ln_1(x)
        attn_out, weights = self.attn(normed_x, return_weights=return_weights)
        x = x + attn_out

        x = x + self.mlp(self.ln_2(x))
        return x, weights

"""
Transformer Block implementation from scratch for MiniGPT.
Follows the modern Pre-LayerNorm GPT architecture with residual connections and GELU activation.
"""
import torch
import torch.nn as nn
from scratch.attention import CausalSelfAttentionScratch


class FeedForwardNetwork(nn.Module):
    """
    MLP / Feed-Forward Network: expands hidden dimension by 4x with GELU activation.
    FFN(x) = Linear(4 * d, d)(GELU(Linear(d, 4 * d)(x)))
    """
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlockScratch(nn.Module):
    """
    Pre-LN Transformer Block:
      x = x + Attention(LayerNorm(x))
      x = x + MLP(LayerNorm(x))
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
        self.attn = CausalSelfAttentionScratch(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardNetwork(embed_dim=embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Causal Self-Attention with residual connection
        attn_out, _ = self.attn(self.ln_1(x))
        x = x + attn_out
        # Pre-LN MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


TransformerBlock = TransformerBlockScratch


if __name__ == "__main__":
    B, T, C = 2, 16, 64
    x = torch.randn(B, T, C)
    block = TransformerBlockScratch(embed_dim=C, num_heads=4, context_length=32)
    out = block(x)

    print("Transformer Block input shape: ", x.shape)
    print("Transformer Block output shape:", out.shape)
    assert out.shape == (B, T, C)
    assert not torch.isnan(out).any()
    print("Transformer Block scratch verified successfully!")

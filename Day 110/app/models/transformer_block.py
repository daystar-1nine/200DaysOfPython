"""
PyTorch Transformer Encoder Block module.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from app.models.multi_head_attention import MultiHeadAttention
from app.models.feed_forward import PositionwiseFeedForward


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Layer (Post-LN or Pre-LN).
    Supports ablation configurations:
        - use_residual: whether to apply skip-connections
        - use_layer_norm: whether to apply LayerNorm
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)

        self.norm1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional boolean padding mask
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Multi-Head Attention Sub-layer
        attn_out, weights = self.self_attention(x, x, x, mask=mask)
        attn_out = self.dropout1(attn_out)

        if self.use_residual:
            x = x + attn_out
        else:
            x = attn_out
        x = self.norm1(x)

        # Feed-Forward Sub-layer
        ffn_out = self.feed_forward(x)
        ffn_out = self.dropout2(ffn_out)

        if self.use_residual:
            x = x + ffn_out
        else:
            x = ffn_out
        x = self.norm2(x)

        return x, weights

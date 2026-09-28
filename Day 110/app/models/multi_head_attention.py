"""
PyTorch Multi-Head Attention layer.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from app.models.attention import scaled_dot_product_attention_torch


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism according to 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (batch, seq_len_q, d_model)
            k: (batch, seq_len_k, d_model)
            v: (batch, seq_len_v, d_model)
            mask: Optional boolean mask (True at padding positions)
        Returns:
            output: (batch, seq_len_q, d_model)
            weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        q_heads = self._split_heads(q_proj)
        k_heads = self._split_heads(k_proj)
        v_heads = self._split_heads(v_proj)

        context, weights = scaled_dot_product_attention_torch(
            q_heads,
            k_heads,
            v_heads,
            mask=mask,
            dropout_p=self.dropout,
            training=self.training
        )

        concat = self._combine_heads(context)
        output = self.out_proj(concat)
        return output, weights

"""
Token Embedding and Positional Encoding module for PyTorch.
"""
import math
import torch
import torch.nn as nn
from app.preprocessing.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    Combines vocabulary lookup embedding with sinusoidal positional encoding.
    Multiplies token embeddings by sqrt(d_model) following Vaswani et al.
    Supports ablation: use_positional_encoding=False.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)

        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        embeddings = self.token_embedding(token_ids) * self.scale
        if self.use_positional_encoding:
            return self.pos_encoder(embeddings)
        return self.dropout(embeddings)

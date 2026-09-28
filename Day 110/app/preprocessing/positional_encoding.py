"""
Positional Encoding layer and generator for PyTorch models.
"""
from typing import Optional
import math
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding module for PyTorch Transformer architectures.
    Adds position information to input embeddings: x + PE(pos).
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def get_positional_encoding_numpy(max_len: int, d_model: int) -> np.ndarray:
    """Helper to retrieve sinusoidal positional encoding table as a NumPy array."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    positions = np.arange(max_len)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dimensions // 2)) / d_model)
    angles = positions * angle_rates
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

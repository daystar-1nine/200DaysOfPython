"""
PyTorch Position-wise Feed-Forward Network layer.
"""
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

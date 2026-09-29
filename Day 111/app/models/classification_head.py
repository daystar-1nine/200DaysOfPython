"""
Classification head module for BERT-based sentence representation.
"""
import torch
import torch.nn as nn


class BertClassificationHead(nn.Module):
    """
    Standard classification head applied to the [CLS] sequence representation.
    Pooler: Dropout -> Linear(hidden_size, hidden_dim) -> Activation -> Dropout -> Linear(hidden_dim, 1) -> Sigmoid.
    """
    def __init__(self, hidden_size: int = 128, num_classes: int = 1, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, hidden_size) — [CLS] token representation
        Returns:
            probs: Tensor of shape (batch_size, 1) in range [0, 1]
        """
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return self.sigmoid(logits)

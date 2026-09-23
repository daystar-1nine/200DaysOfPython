"""
Neural text classifier module for Day 105: Neural NLP & Text Classification.
Implements trainable embedding layer, masked pooling, and dense classification head.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from ..embeddings.pooling import masked_global_average_pooling


class NeuralTextClassifier(nn.Module):
    """Feedforward neural text classifier with trainable embedding layer and masked pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: Optional[int] = 64,
        dropout_rate: float = 0.5,
        pad_id: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.pad_id = pad_id

        # 1. Trainable Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        # 2. Hidden Layer(s)
        if hidden_dim is not None and hidden_dim > 0:
            self.has_hidden = True
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.has_hidden = False
            self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with integer token IDs.
            mask: Optional binary tensor of shape (batch_size, seq_len).
            
        Returns:
            Logits tensor of shape (batch_size,).
        """
        # (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)

        if mask is not None:
            # (batch_size, embedding_dim)
            pooled = masked_global_average_pooling(embeddings, mask)
        else:
            # Fallback to standard mean pooling if no mask provided
            pooled = embeddings.mean(dim=1)

        if self.has_hidden:
            h = self.relu(self.fc1(pooled))
            h = self.dropout(h)
            logits = self.classifier(h)
        else:
            logits = self.classifier(pooled)

        return logits.squeeze(-1)

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute sigmoid class probabilities for spam (class 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, mask=mask)
            return torch.sigmoid(logits)

    def count_parameters(self) -> Dict[str, int]:
        """Calculate and return breakdown of model parameter counts."""
        emb_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
        dense_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "embedding" not in name and p.requires_grad
        )
        total_params = emb_params + dense_params

        return {
            "embedding_params": emb_params,
            "dense_params": dense_params,
            "total_params": total_params
        }

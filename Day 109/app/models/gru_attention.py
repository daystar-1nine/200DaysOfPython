import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import AttentionLayer

class GRUAttentionClassifier(nn.Module):
    """
    GRU with Attention Classifier matching Section 17 & 20:
      Input (batch, seq_len)
        ↓
      Embedding(vocab_size, embedding_dim, padding_idx=0)
        ↓
      GRU(hidden_dim, batch_first=True) with return_sequences=True
        ↓
      AttentionLayer(hidden_dim, attention_dim) -> (batch, hidden_dim) & (batch, seq_len)
        ↓
      Dropout(0.3)
        ↓
      Dense(hidden_dim -> 32, ReLU)
        ↓
      Dense(32 -> 1, Sigmoid)
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dense_dim: int = 32,
        dropout_rate: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.attention = AttentionLayer(hidden_dim=hidden_dim, attention_dim=attention_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        """
        x: (batch, seq_len)
        mask: Optional (batch, seq_len)
        """
        if mask is None:
            # Auto-generate mask from padding_idx (1 for valid, 0 for pad)
            mask = (x != self.pad_idx).long()
            
        embedded = self.embedding(x)
        # Full sequence of hidden states: (batch, seq_len, hidden_dim)
        seq_hidden, _ = self.gru(embedded)
        
        # Apply mask-aware attention
        context, attn_weights = self.attention(seq_hidden, mask=mask)
        
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        probs = self.sigmoid(out).squeeze(-1)
        
        if return_attention:
            return probs, attn_weights
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

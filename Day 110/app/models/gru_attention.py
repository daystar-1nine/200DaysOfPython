"""
PyTorch GRU with Attention Classifier for direct Day 109 baseline comparison.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        attn_dim: int = 64,
        pad_idx: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.w_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.pad_idx = pad_idx

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = (x == self.pad_idx)

        embeds = self.embedding(x)
        gru_out, _ = self.gru(embeds)  # (B, L, H)

        scores = self.v_proj(torch.tanh(self.w_proj(gru_out))).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(mask.bool(), float("-1e9"))
        weights = F.softmax(scores, dim=-1)  # (B, L)

        context = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)  # (B, H)
        probs = self.classifier(context)
        return probs, weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

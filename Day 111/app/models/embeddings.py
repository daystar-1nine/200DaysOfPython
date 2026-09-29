"""
PyTorch module for BERT Input Embeddings: Token + Position + Segment Embeddings.
"""
from typing import Optional
import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """
    Constructs the complete BERT input representation:
        Embedding = TokenEmbedding + PositionEmbedding + SegmentEmbedding
    Followed by LayerNorm and Dropout.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 128,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout_prob: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        batch_size = input_ids.size(0)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

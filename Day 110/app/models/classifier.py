"""
Mini Transformer SMS Spam Classifier implementation in PyTorch.
"""
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from app.models.embedding import TransformerEmbedding
from app.models.transformer_encoder import TransformerEncoder


class MiniTransformerClassifier(nn.Module):
    """
    Mini Transformer Classifier for SMS Spam detection.
    Architecture:
        Token IDs -> Embedding (+ Positional Encoding) -> N x TransformerEncoderBlock
        -> Mask-aware Average Pooling -> Dense(64) -> Dropout -> Dense(1) -> Sigmoid.
    Supports complete ablation toggles.
    """
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 100,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
        pooling: str = "mask_mean",
        use_positional_encoding: bool = True,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        use_padding_mask: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.pooling = pooling

        # Ablation flags
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_padding_mask = use_padding_mask

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_length + 20,
            dropout=dropout,
            pad_idx=pad_idx,
            use_positional_encoding=use_positional_encoding
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation="relu",
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )

        self.pooler = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            token_ids: (batch_size, seq_len)
            mask: Optional boolean padding mask (True at padding positions)
        Returns:
            probs: (batch_size, 1) prediction probabilities
            all_weights: List of attention matrices per layer
        """
        if mask is None and self.use_padding_mask:
            mask = (token_ids == self.pad_idx)
        elif not self.use_padding_mask:
            mask = None

        # 1. Embedding + Positional Encoding
        x = self.embedding(token_ids)

        # 2. Transformer Encoder Stack
        x, all_weights = self.encoder(x, mask=mask)

        # 3. Sequence Pooling
        if self.pooling == "mask_mean" and mask is not None:
            # Mask-aware average pooling: ignore padding tokens
            valid_mask = (~mask).unsqueeze(-1).float()  # (B, L, 1)
            sum_embeddings = (x * valid_mask).sum(dim=1)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / valid_counts
        elif self.pooling == "first":
            pooled = x[:, 0, :]
        else:
            # Global average pooling
            pooled = x.mean(dim=1)

        # 4. Dense classification head
        logits = self.pooler(pooled)
        probs = torch.sigmoid(logits)
        return probs, all_weights

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

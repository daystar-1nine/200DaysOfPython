"""
Stack of Transformer Encoder Blocks in PyTorch.
"""
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from app.models.transformer_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    """
    Encoder stack consisting of N identical TransformerEncoderBlock layers.
    Collects attention matrices across all layers for analysis and visualization.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional padding mask
        Returns:
            output: (batch_size, seq_len, d_model)
            all_weights: List of length num_layers with shape (batch_size, num_heads, seq_len, seq_len)
        """
        all_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask=mask)
            all_weights.append(weights)
        return x, all_weights

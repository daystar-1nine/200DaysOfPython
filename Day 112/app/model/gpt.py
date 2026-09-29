"""
Production MiniGPT Model Architecture.
Decoder-only autoregressive language model with causal multi-head self-attention.
"""
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model.block import TransformerBlock


class MiniGPT(nn.Module):
    """
    MiniGPT: Autoregressive Decoder-Only Transformer Language Model.
    Predicts the next token probability distribution conditioned on preceding context:
        P(x_1, ..., x_T) = prod_{t=1}^T P(x_t | x_1, ..., x_{t-1})
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=context_length,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language Modeling Projection Head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight Tying
        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            targets: Optional LongTensor of shape (batch_size, seq_len)
            return_attentions: Whether to collect self-attention matrices
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
            loss: Scalar cross-entropy loss if targets is given, else None
            attentions: List of attention matrices per layer if requested
        """
        device = input_ids.device
        b, t = input_ids.size()
        if t > self.context_length:
            raise ValueError(
                f"Input sequence length ({t}) exceeds context window capacity ({self.context_length})"
            )

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(pos)
        x = self.dropout(tok_emb + pos_emb)

        attentions = [] if return_attentions else None
        for block in self.blocks:
            x, weights = block(x, return_weights=return_attentions)
            if return_attentions:
                attentions.append(weights)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss, attentions

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

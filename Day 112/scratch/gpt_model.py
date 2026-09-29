"""
MiniGPT complete model architecture from scratch.
Decoder-only autoregressive Transformer with learned positional embeddings and causal attention.
"""
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from scratch.transformer_block import TransformerBlockScratch


class MiniGPTScratch(nn.Module):
    """
    Miniature Generative Pretrained Transformer (GPT) model.
    Predicts next token probabilities autoregressively given previous context.
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

        # Embedding tables
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockScratch(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=context_length,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language Modeling Head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017; Radford et al., 2019)
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # Initialize weights
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
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            targets: Optional tensor of shape (batch_size, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy scalar loss if targets provided, else None
        """
        device = input_ids.device
        b, t = input_ids.size()
        if t > self.context_length:
            raise ValueError(f"Sequence length ({t}) exceeds context window capacity ({self.context_length})")

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.tok_embeddings(input_ids)      # (b, t, embed_dim)
        pos_emb = self.pos_embeddings(pos)            # (t, embed_dim)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                      # (b, t, vocab_size)

        loss = None
        if targets is not None:
            # Flatten across batch and sequence dimensions for cross-entropy
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


MiniGPT = MiniGPTScratch


if __name__ == "__main__":
    vocab_size = 65
    model = MiniGPTScratch(vocab_size=vocab_size, context_length=32, embed_dim=64, num_heads=4, num_layers=2)
    x = torch.randint(0, vocab_size, (2, 16))
    y = torch.randint(0, vocab_size, (2, 16))

    logits, loss = model(x, targets=y)
    print("Logits shape:", logits.shape)
    print("Loss value:  ", loss.item())
    assert logits.shape == (2, 16, vocab_size)
    assert loss is not None and loss.item() > 0.0

    params = model.count_parameters()
    print("Parameter summary:", params)
    print("MiniGPT scratch verified successfully!")

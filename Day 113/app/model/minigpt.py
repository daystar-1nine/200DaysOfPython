"""
Scalable MiniGPT Architecture with Pre-Layer Normalization, GELU MLP,
causal attention masking, weight tying, and mixed precision support.
"""
import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Projects inputs into Q, K, V and applies lower-triangular causal mask:
        M_ij = 0.0 for j <= i, -inf for j > i
    """
    def __init__(self, embed_dim: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Fused projection for Q, K, V
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length))
                 .view(1, 1, context_length, context_length)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.size()

        # Compute Q, K, V in single batched GEMM
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, H, T, d_k)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att, dim=-1)
        att_dropped = self.attn_dropout(att_weights)

        y = att_dropped @ v  # (B, H, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(y))

        return out, (att_weights if return_weights else None)


class GELUMLP(nn.Module):
    """Feed-Forward Network with GELU non-linearity and 4x expansion."""
    def __init__(self, embed_dim: int, dropout: float = 0.1, ff_multiplier: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, ff_multiplier * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ff_multiplier * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gelu(self.c_fc(x))
        out = self.dropout(self.c_proj(h))
        return out


class TransformerBlock(nn.Module):
    """
    Pre-Layer Normalization Transformer Block:
        x = x + Attn(LN1(x))
        x = x + MLP(LN2(x))
    """
    def __init__(self, embed_dim: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, context_length, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = GELUMLP(embed_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, weights = self.attn(self.ln_1(x), return_weights=return_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class ScalableMiniGPT(nn.Module):
    """
    Decoder-Only Transformer Language Model supporting configurable scaling tiers
    (Tiny, Small, Medium) and weight tying.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        # Embeddings
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embeddings = nn.Embedding(self.context_length, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                context_length=self.context_length,
                dropout=config.dropout
            )
            for _ in range(self.num_layers)
        ])

        # Final LayerNorm and Unembedding
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Weight tying: share embedding weights with unembedding projection
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Dict[str, int]:
        """Calculates exact total and component parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embed_params = self.token_embeddings.weight.numel() + self.position_embeddings.weight.numel()
        blocks_params = sum(p.numel() for b in self.blocks for p in b.parameters())
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters())

        # If tied, lm_head adds 0 unique parameters
        lm_head_params = 0 if self.config.tie_weights else self.lm_head.weight.numel()

        return {
            "total": total,
            "trainable": trainable,
            "embeddings": embed_params,
            "blocks": blocks_params,
            "final_layernorm": ln_f_params,
            "lm_head": lm_head_params
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Args:
            idx: LongTensor of token IDs of shape (batch_size, seq_len)
            targets: Optional shifted target LongTensor of shape (batch_size, seq_len)
            return_weights: Whether to collect and return attention weight maps
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy scalar loss (None if targets is None)
            all_weights: Optional list of attention weight tensors from each block
        """
        device = idx.device
        B, T = idx.size()

        if T > self.context_length:
            raise ValueError(f"Sequence length ({T}) exceeds context window ({self.context_length})")

        # Position indices: [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.token_embeddings(idx)  # (B, T, C)
        pos_emb = self.position_embeddings(pos)  # (T, C)
        x = self.dropout(tok_emb + pos_emb)

        all_weights = [] if return_weights else None
        for block in self.blocks:
            x, w = block(x, return_weights=return_weights)
            if return_weights:
                all_weights.append(w)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, all_weights

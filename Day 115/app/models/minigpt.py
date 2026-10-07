"""
Transformer backbone architecture for Day 115: Preference Optimization & RLHF.
Implements Pre-LayerNorm Transformer blocks, Causal Multi-Head Self-Attention,
and weight-tied Language Model head.
"""
import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with causal mask and attention dropout."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        # Projections
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask lower triangular [1, 1, context_length, context_length]
        causal_mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()

        # Compute query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape to [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Causal mask
        causal = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(causal == 0, float("-inf"))

        # Padding attention mask [B, 1, 1, T]
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask.view(B, 1, 1, T)
            else:
                pad_mask = attention_mask
            att = att.masked_fill(pad_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        # Avoid NaN from all-masked rows
        att = torch.nan_to_num(att, nan=0.0)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class GELUMLP(nn.Module):
    """Feed-Forward Network with GELU activation and 4x hidden expansion."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = GELUMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPTBackbone(nn.Module):
    """
    Core Transformer backbone that outputs hidden representations H in R^{B x T x D}.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes contextualized hidden representations.
        Returns:
          hidden_states: [B, T, D]
        """
        B, T = input_ids.size()
        if T > self.config.context_length:
            raise ValueError(f"Sequence length {T} exceeds maximum context length {self.config.context_length}")

        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_vecs = self.tok_emb(input_ids)
        pos_vecs = self.pos_emb(positions)
        x = self.drop(tok_vecs + pos_vecs)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        return self.ln_f(x)

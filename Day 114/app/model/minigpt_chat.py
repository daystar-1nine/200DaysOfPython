"""
MiniGPT-Chat Architecture: Pre-LN Transformer decoder with causal self-attention,
loss masking support, autoregressive generation, and integrated LoRA adapter support.
"""
import math
from typing import Optional, Dict, Any, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config import ChatModelConfig, LoRAConfig


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapping a frozen linear transformation:
      h = x W^T + (alpha / r) * (x A^T B^T)
    Where:
      - W is frozen base weight [out_features, in_features]
      - A is initialized randomly [r, in_features]
      - B is initialized to zero [out_features, r]
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Base linear layer (frozen)
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        if r > 0:
            # Low-rank adapters
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        if self.r == 0 or self.merged:
            return base_out

        # LoRA branch: x @ A^T @ B^T * scaling
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> None:
        """Merges low-rank weights into base weights for zero-overhead inference."""
        if self.r > 0 and not self.merged:
            # W = W + scaling * (B @ A)
            update = (self.lora_B @ self.lora_A) * self.scaling
            self.base_linear.weight.data.add_(update)
            self.merged = True

    def unmerge_weights(self) -> None:
        """Subtracts merged low-rank weights to restore original base weights."""
        if self.r > 0 and self.merged:
            update = (self.lora_B @ self.lora_A) * self.scaling
            self.base_linear.weight.data.sub_(update)
            self.merged = False


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with causal masking and padding mask support.
    """
    def __init__(self, config: ChatModelConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of shape [1, 1, context_length, context_length]
        causal_mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.context_length, config.context_length))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 1. Apply causal mask (future tokens receive -inf)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # 2. Apply padding mask if provided ([B, T] -> [B, 1, 1, T])
        if attention_mask is not None:
            # attention_mask: 1 for valid tokens, 0 for pad
            pad_mask = attention_mask.view(B, 1, 1, T)
            att = att.masked_fill(pad_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        # In case an entire row is masked (e.g. padding rows), replace NaNs with zeros
        att = torch.nan_to_num(att, nan=0.0)
        att = self.attn_dropout(att)

        y = att @ v  # [B, num_heads, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class GELUMLP(nn.Module):
    """Feed-forward network with Gaussian Error Linear Units (GELU)."""
    def __init__(self, config: ChatModelConfig):
        super().__init__()
        d_ff = 4 * config.embed_dim
        self.c_fc = nn.Linear(config.embed_dim, d_ff, bias=True)
        self.c_proj = nn.Linear(d_ff, config.embed_dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""
    def __init__(self, config: ChatModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = GELUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPTChat(nn.Module):
    """
    Complete MiniGPT-Chat architecture supporting full fine-tuning, LoRA adapters,
    and loss masking over prompt tokens.
    """
    def __init__(self, config: ChatModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Parameter initialization
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
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        - input_ids: LongTensor [B, T]
        - attention_mask: LongTensor [B, T] (1 for valid, 0 for pad)
        - targets: LongTensor [B, T] with -100 at positions to ignore in loss
        """
        B, T = input_ids.size()
        if T > self.config.context_length:
            raise ValueError(f"Sequence length {T} exceeds context window {self.config.context_length}")

        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        # Sum token and positional embeddings
        tok_vecs = self.tok_emb(input_ids)
        pos_vecs = self.pos_emb(positions)
        x = self.drop(tok_vecs + pos_vecs)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]

        loss = None
        if targets is not None:
            # Shift tokens: predict next token from previous positions
            # logits at position t predict target at position t+1
            # If target sequence is already aligned (as in SFT collator):
            # Collator input_ids is [x_0, ..., x_{T-1}], targets is [y_0, ..., y_{T-1}]
            # We compute cross_entropy between logits[:, :-1, :] and targets[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_token_id: Optional[int] = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, top-p, and repetition penalty.
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            cond = generated if generated.size(1) <= self.config.context_length else generated[:, -self.config.context_length:]
            logits, _ = self.forward(cond)
            # Focus only on the last step
            next_token_logits = logits[:, -1, :].clone()

            # Apply repetition penalty to recently generated response tokens (not the prompt)
            if repetition_penalty != 1.0:
                prompt_len = prompt_ids.size(1)
                for b in range(generated.size(0)):
                    if generated.size(1) > prompt_len:
                        recent_tokens = set(generated[b, max(prompt_len, generated.size(1) - 20):].tolist())
                        for token_id in recent_tokens:
                            if next_token_logits[b, token_id] > 0:
                                next_token_logits[b, token_id] /= repetition_penalty
                            else:
                                next_token_logits[b, token_id] *= repetition_penalty

            if not do_sample or temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-4)

                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")

                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices right to keep the first token above top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if stop_token_id is not None and (next_token == stop_token_id).all():
                break

        return generated

    def apply_lora(
        self,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Freezes base model parameters and injects LoRALinear layers into target modules.
        Returns parameter count statistics.
        """
        if target_modules is None:
            target_modules = ["c_attn", "c_proj"]

        # Freeze all base model parameters
        for p in self.parameters():
            p.requires_grad = False

        lora_layers_added = 0
        for block_idx, block in enumerate(self.blocks):
            # Self-attention target modules
            if "c_attn" in target_modules and isinstance(block.attn.c_attn, nn.Linear):
                block.attn.c_attn = LoRALinear(block.attn.c_attn, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                lora_layers_added += 1
            if "c_proj" in target_modules and isinstance(block.attn.c_proj, nn.Linear):
                block.attn.c_proj = LoRALinear(block.attn.c_proj, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                lora_layers_added += 1

        stats = self.count_parameters()
        stats["lora_layers_added"] = lora_layers_added
        return stats

    def count_parameters(self) -> Dict[str, int]:
        """Returns total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_percent": round((trainable / max(total, 1)) * 100, 2)
        }

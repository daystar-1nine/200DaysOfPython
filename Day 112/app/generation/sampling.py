"""
Production sampling strategies for MiniGPT:
- Temperature scaling
- Top-k truncation
- Top-p (nucleus) thresholding
- Combined sampling pipeline
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.tokenizer.char_tokenizer import CharacterTokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Samples next token IDs given logits from the final time step.

    Args:
        logits: FloatTensor of shape (batch_size, vocab_size)
        temperature: Temperature scaling factor. If <= 0, applies greedy argmax.
        top_k: Number of highest probability vocabulary tokens to retain.
        top_p: Nucleus cumulative probability threshold.
    Returns:
        LongTensor of shape (batch_size, 1)
    """
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 1. Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # 2. Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        min_topk = topk_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)

    # 3. Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate_with_strategy(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    context_length: Optional[int] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Autoregressively generates text from a prompt using specified sampling strategy.
    """
    model.eval()
    ctx_len = context_length or getattr(model, "context_length", 64)

    if len(prompt) == 0:
        prompt_ids = torch.tensor([[0]], dtype=torch.long, device=device)
    else:
        encoded = tokenizer.encode(prompt)
        prompt_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    idx = prompt_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ctx_len:]
        logits, _, _ = model(idx_cond)
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        idx = torch.cat([idx, next_token], dim=1)

    generated_tokens = idx[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)

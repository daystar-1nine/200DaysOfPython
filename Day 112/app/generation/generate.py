"""
Autoregressive text generation engine for MiniGPT.
"""
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from app.tokenizer.char_tokenizer import CharacterTokenizer


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int
) -> torch.Tensor:
    """
    Iterative autoregressive token generation with context window sliding.

    Args:
        model: MiniGPT language model
        input_ids: Tensor of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        context_length: Model context length capacity
    Returns:
        Tensor of shape (batch_size, seq_len + max_new_tokens)
    """
    model.eval()
    idx = input_ids.clone()

    for _ in range(max_new_tokens):
        # Crop context if longer than model capacity
        idx_cond = idx[:, -context_length:]

        # Forward pass
        logits, _, _ = model(idx_cond)

        # Pluck logits at final time step: (B, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Greedy argmax selection
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append to ongoing generation
        idx = torch.cat([idx, next_token], dim=1)

    return idx


@torch.no_grad()
def generate_text_from_prompt(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    context_length: int = 64,
    device: Optional[torch.device] = None
) -> str:
    """
    Encodes prompt, runs autoregressive generation, and decodes back to text.
    """
    if len(prompt) == 0:
        # If prompt is empty, start from index 0 or random token
        prompt_ids = torch.tensor([[0]], dtype=torch.long, device=device)
    else:
        encoded = tokenizer.encode(prompt)
        prompt_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    out_ids = generate_tokens(
        model=model,
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length
    )

    generated_tokens = out_ids[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)

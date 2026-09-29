"""
Autoregressive generation loop from scratch.
Implements iterative next-token prediction with context window truncation and greedy decoding.
"""
from typing import Optional
import torch
import torch.nn as nn


@torch.no_grad()
def generate_greedy_scratch(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int
) -> torch.Tensor:
    """
    Autoregressively generates new tokens using greedy argmax decoding.

    Args:
        model: MiniGPT language model
        prompt_ids: LongTensor of shape (batch_size, prompt_len)
        max_new_tokens: Number of tokens to append
        context_length: Maximum context window capacity
    Returns:
        Tensor of shape (batch_size, prompt_len + max_new_tokens)
    """
    model.eval()
    idx = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate to most recent context window
        idx_cond = idx[:, -context_length:]

        # Forward pass
        logits = model(idx_cond)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Extract logits at the final position: (B, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Greedy selection: pick token with maximum logit
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Concatenate to sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx


if __name__ == "__main__":
    from scratch.gpt_model import MiniGPTScratch

    vocab_size = 50
    model = MiniGPTScratch(vocab_size=vocab_size, context_length=16, embed_dim=32, num_heads=2, num_layers=2)
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = generate_greedy_scratch(model, prompt, max_new_tokens=10, context_length=16)

    print("Prompt shape:   ", prompt.shape)
    print("Generated shape:", generated.shape)
    assert generated.shape == (1, 15)
    assert (generated[:, :5] == prompt).all(), "Prompt prefix must be preserved!"
    print("Greedy autoregressive generation scratch verified successfully!")

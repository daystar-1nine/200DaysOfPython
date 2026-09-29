"""
Day 112 - Coding Challenge 5: Autoregressive Next-Token Generation Loop
Problem: Implement autoregressive text generation with context length sliding window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


def generate_greedy(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_length: int
) -> torch.Tensor:
    """
    Autoregressively generates max_new_tokens using greedy decoding (argmax).
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context if it exceeds model capacity
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        with torch.no_grad():
            logits, _, _ = model(idx_cond)
            # Take logits at last position: (B, V)
            logits_last = logits[:, -1, :]
            next_token = torch.argmax(logits_last, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
    return idx


if __name__ == "__main__":
    from app.model.gpt import MiniGPT
    m = MiniGPT(vocab_size=20, context_length=8, embed_dim=16, num_heads=2, num_layers=1)
    prompt = torch.tensor([[1, 2, 3]])
    generated = generate_greedy(m, prompt, max_new_tokens=6, context_length=8)
    assert generated.shape == (1, 9)
    assert (generated[:, :3] == prompt).all()
    print("Challenge 5: PASSED")

"""
Day 112 - Coding Challenge 8: Top-p (Nucleus) Sampling
Problem: Filter smallest set of tokens whose cumulative probability mass exceeds threshold p.
"""
import torch
import torch.nn.functional as F


def sample_top_p(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above p (shift right by 1 to keep first token above p)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    filtered_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


if __name__ == "__main__":
    # Logits where index 0 has >90% probability
    logits = torch.tensor([[10.0, 2.0, 0.0, -5.0]])
    for _ in range(25):
        token = sample_top_p(logits, p=0.5, temperature=1.0)
        assert token.item() == 0
    print("Challenge 8: PASSED")

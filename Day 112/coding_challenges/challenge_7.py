"""
Day 112 - Coding Challenge 7: Top-k Sampling
Problem: Filter logits to retain only the top-k highest scoring tokens, masking others with -inf.
"""
import torch
import torch.nn.functional as F


def sample_top_k(logits: torch.Tensor, k: int = 5, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    v, _ = torch.topk(scaled_logits, min(k, scaled_logits.size(-1)))
    # Any logit smaller than the k-th highest logit is set to -inf
    min_val = v[:, [-1]]
    filtered_logits = torch.where(scaled_logits < min_val, float("-inf"), scaled_logits)
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


if __name__ == "__main__":
    logits = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    for _ in range(25):
        token = sample_top_k(logits, k=2, temperature=1.0)
        assert token.item() in (2, 3)
    print("Challenge 7: PASSED")

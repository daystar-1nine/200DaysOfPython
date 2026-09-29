"""
Day 112 - Coding Challenge 9: Perplexity Computation
Problem: Implement perplexity from cross-entropy loss with overflow clamping:
         PPL = exp(loss)
"""
import math
import torch
import torch.nn.functional as F


def calculate_perplexity(loss: float, max_loss: float = 50.0) -> float:
    clamped_loss = min(float(loss), max_loss)
    return round(math.exp(clamped_loss), 4)


def perplexity_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    B, T, V = logits.size()
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
    return calculate_perplexity(loss.item())


if __name__ == "__main__":
    assert calculate_perplexity(0.0) == 1.0
    assert math.isclose(calculate_perplexity(1.0), 2.7183, rel_tol=1e-3)
    assert not math.isinf(calculate_perplexity(1000.0))

    # Perfect prediction test
    logits = torch.zeros(2, 4, 10)
    targets = torch.tensor([[1, 2, 3, 4], [0, 9, 8, 7]])
    for b in range(2):
        for t in range(4):
            logits[b, t, targets[b, t]] = 50.0
    ppl = perplexity_from_logits(logits, targets)
    assert math.isclose(ppl, 1.0, abs_tol=1e-3)
    print("Challenge 9: PASSED")

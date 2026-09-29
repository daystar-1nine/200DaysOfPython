"""
Day 112 - Coding Challenge 6: Temperature Scaling Sampling
Problem: Implement temperature scaling for next-token sampling:
         P(x_i) = softmax(logits / T)
"""
import numpy as np
import torch
import torch.nn.functional as F


def sample_temperature_numpy(logits: np.ndarray, temperature: float = 1.0) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs))


def sample_temperature_torch(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


if __name__ == "__main__":
    logits_np = np.array([10.0, 1.0, 0.0])
    # Low temp -> near deterministic
    pick = sample_temperature_numpy(logits_np, temperature=0.01)
    assert pick == 0

    logits_torch = torch.tensor([[10.0, 1.0, 0.0]])
    pick_t = sample_temperature_torch(logits_torch, temperature=0.01)
    assert pick_t.item() == 0
    print("Challenge 6: PASSED")

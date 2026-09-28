"""
Challenge 6: Calculate attention entropy.
"""
import numpy as np

def calculate_attention_entropy(weights: np.ndarray, eps: float = 1e-9) -> float:
    """
    H(A) = - sum_i A_i * log(A_i + eps)
    """
    weights = np.asarray(weights)
    weights = weights / np.maximum(np.sum(weights), eps) # normalize
    entropy = -np.sum(weights * np.log(weights + eps))
    return float(entropy)

if __name__ == "__main__":
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    h_uniform = calculate_attention_entropy(uniform)
    h_peaked = calculate_attention_entropy(peaked)
    print(f"Uniform entropy: {h_uniform:.4f} | Peaked entropy: {h_peaked:.4f}")
    assert h_uniform > h_peaked
    print("Challenge 6 passed!")

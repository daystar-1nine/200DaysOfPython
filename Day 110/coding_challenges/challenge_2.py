"""
Challenge 2: Implement scaled dot product attention with masking.
"""
from typing import Optional, Tuple
import numpy as np


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = np.matmul(weights, v)
    return context, weights


if __name__ == "__main__":
    q = np.random.randn(2, 4, 16)
    k = np.random.randn(2, 4, 16)
    v = np.random.randn(2, 4, 16)
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[:, :, 3] = True  # mask out 4th token

    ctx, w = scaled_dot_product_attention(q, k, v, mask=mask)
    print("Context shape:", ctx.shape, "Weights shape:", w.shape)
    assert ctx.shape == (2, 4, 16)
    assert np.allclose(w[:, :, 3], 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)
    print("Challenge 2 passed!")

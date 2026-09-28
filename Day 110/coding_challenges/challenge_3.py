"""
Challenge 3: Implement MultiHeadAttention from scratch.
"""
from typing import Optional, Tuple
import numpy as np


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.w_q = rng.randn(d_model, d_model) * scale
        self.w_k = rng.randn(d_model, d_model) * scale
        self.w_v = rng.randn(d_model, d_model) * scale
        self.w_o = rng.randn(d_model, d_model) * scale

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None):
        b, l_q, _ = q.shape
        _, l_k, _ = k.shape

        q_proj = np.matmul(q, self.w_q).reshape(b, l_q, self.num_heads, self.d_k).swapaxes(1, 2)
        k_proj = np.matmul(k, self.w_k).reshape(b, l_k, self.num_heads, self.d_k).swapaxes(1, 2)
        v_proj = np.matmul(v, self.w_v).reshape(b, l_k, self.num_heads, self.d_k).swapaxes(1, 2)

        scores = np.matmul(q_proj, k_proj.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(mask, -1e9, scores)

        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        context = np.matmul(weights, v_proj).swapaxes(1, 2).reshape(b, l_q, self.d_model)
        output = np.matmul(context, self.w_o)
        return output, weights


if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    x = np.random.randn(2, 6, 64)
    out, w = mha.forward(x, x, x)
    print("MHA out shape:", out.shape, "Weights shape:", w.shape)
    assert out.shape == (2, 6, 64)
    assert w.shape == (2, 4, 6, 6)
    print("Challenge 3 passed!")

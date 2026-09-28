"""
Challenge 5: Implement TransformerBlock without using a prebuilt library block.
"""
from typing import Optional, Tuple
import numpy as np


class MiniTransformerBlock:
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128, seed: int = 42):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.RandomState(seed)
        scale_mha = np.sqrt(2.0 / (d_model + self.d_k))
        self.w_q = rng.randn(d_model, d_model) * scale_mha
        self.w_k = rng.randn(d_model, d_model) * scale_mha
        self.w_v = rng.randn(d_model, d_model) * scale_mha
        self.w_o = rng.randn(d_model, d_model) * scale_mha

        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.w1 = rng.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.w2 = rng.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def _norm(self, x, eps=1e-5):
        m = np.mean(x, axis=-1, keepdims=True)
        v = np.var(x, axis=-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        b, l, _ = x.shape
        # Attention
        q = np.matmul(x, self.w_q).reshape(b, l, self.num_heads, self.d_k).swapaxes(1, 2)
        k = np.matmul(x, self.w_k).reshape(b, l, self.num_heads, self.d_k).swapaxes(1, 2)
        v = np.matmul(x, self.w_v).reshape(b, l, self.num_heads, self.d_k).swapaxes(1, 2)

        scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(mask, -1e9, scores)
        s_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - s_max)
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        ctx = np.matmul(weights, v).swapaxes(1, 2).reshape(b, l, self.d_model)
        attn_out = np.matmul(ctx, self.w_o)

        # Residual + Norm
        x1 = self._norm(x + attn_out)

        # FFN
        ffn_out = np.matmul(np.maximum(0, np.matmul(x1, self.w1) + self.b1), self.w2) + self.b2
        x2 = self._norm(x1 + ffn_out)
        return x2, weights


if __name__ == "__main__":
    tb = MiniTransformerBlock(64, 4, 128)
    x = np.random.randn(2, 5, 64)
    out, w = tb.forward(x)
    print("Block out shape:", out.shape, "Weights:", w.shape)
    assert out.shape == (2, 5, 64)
    assert w.shape == (2, 4, 5, 5)
    print("Challenge 5 passed!")

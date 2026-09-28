"""
Multi-Head Attention implemented from scratch in pure NumPy.
"""
from typing import Optional, Tuple
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

    Args:
        q: (..., seq_len_q, d_k)
        k: (..., seq_len_k, d_k)
        v: (..., seq_len_v, d_k)
        mask: optional boolean mask where True indicates positions to mask out.
    Returns:
        context: (..., seq_len_q, d_k)
        weights: (..., seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        # Mask positions with very large negative number
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)
    context = np.matmul(weights, v)
    return context, weights


class MultiHeadAttention:
    """
    Multi-Head Attention module in NumPy.

    Splits d_model into num_heads subspaces of dimension d_k = d_model // num_heads.
    Computes parallel scaled dot-product attention across all heads and projects back to d_model.
    """
    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.RandomState(seed)
        # Xavier/Glorot normal initialization scale
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.w_q = rng.randn(d_model, d_model) * scale
        self.w_k = rng.randn(d_model, d_model) * scale
        self.w_v = rng.randn(d_model, d_model) * scale
        self.w_o = rng.randn(d_model, d_model) * scale

        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return np.swapaxes(x, 1, 2)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = np.swapaxes(x, 1, 2)
        return x.reshape(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            q: (batch_size, seq_len_q, d_model)
            k: (batch_size, seq_len_k, d_model)
            v: (batch_size, seq_len_v, d_model)
            mask: optional mask of shape broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
        Returns:
            output: (batch_size, seq_len_q, d_model)
            weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # Linear projections
        q_proj = np.matmul(q, self.w_q) + self.b_q
        k_proj = np.matmul(k, self.w_k) + self.b_k
        v_proj = np.matmul(v, self.w_v) + self.b_v

        # Split into multiple heads
        q_heads = self._split_heads(q_proj)
        k_heads = self._split_heads(k_proj)
        v_heads = self._split_heads(v_proj)

        # Reshape mask if 2D (batch, seq_len_k) or 3D
        if mask is not None:
            if mask.ndim == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask[:, np.newaxis, np.newaxis, :]
            elif mask.ndim == 3:
                mask = mask[:, np.newaxis, :, :]

        # Scaled dot product attention
        context_heads, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask=mask)

        # Concatenate heads and project output
        concat = self._combine_heads(context_heads)
        output = np.matmul(concat, self.w_o) + self.b_o
        return output, weights


if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=128, num_heads=4)
    x = np.random.randn(2, 10, 128)
    out, weights = mha.forward(x, x, x)
    print("MHA output shape:", out.shape)
    print("Attention weights shape:", weights.shape)
    assert out.shape == (2, 10, 128)
    assert weights.shape == (2, 4, 10, 10)
    print("MultiHeadAttention pure NumPy verification successful!")

"""
Layer Normalization implemented from scratch in pure NumPy.
"""
from typing import Optional
import numpy as np


class LayerNorm:
    """
    Layer Normalization:
        y = gamma * ((x - mean) / sqrt(var + eps)) + beta
    Normalizes across the feature dimension for each token independently.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        # Trainable affine parameters
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input array of shape (..., d_model)
        Returns:
            Normalized array of same shape.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


if __name__ == "__main__":
    ln = LayerNorm(d_model=128)
    x = np.random.randn(2, 5, 128) * 10.0 + 5.0
    out = ln.forward(x)
    print("Mean before LN:", np.mean(x, axis=-1)[0, :3])
    print("Variance before LN:", np.var(x, axis=-1)[0, :3])
    print("Mean after LN:", np.mean(out, axis=-1)[0, :3])
    print("Variance after LN:", np.var(out, axis=-1)[0, :3])
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    assert np.allclose(np.var(out, axis=-1), 1.0, atol=1e-3)
    print("NumPy LayerNorm verification successful!")

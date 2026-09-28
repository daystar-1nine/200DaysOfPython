"""
Position-wise Feed-Forward Network implemented from scratch in pure NumPy.
"""
from typing import Optional
import numpy as np


class FeedForward:
    """
    Position-wise Feed-Forward Network (FFN):
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, activation: str = "relu", seed: int = 42):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation.lower()

        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))

        self.w1 = rng.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.w2 = rng.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "gelu":
            # Approximate GELU formulation
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Tensor of shape (..., d_model)
        Returns:
            Tensor of shape (..., d_model)
        """
        hidden = np.matmul(x, self.w1) + self.b1
        activated = self._activate(hidden)
        output = np.matmul(activated, self.w2) + self.b2
        return output


if __name__ == "__main__":
    ffn = FeedForward(d_model=128, d_ff=256)
    x = np.random.randn(2, 10, 128)
    out = ffn.forward(x)
    print("FFN input shape:", x.shape)
    print("FFN output shape:", out.shape)
    assert out.shape == x.shape
    print("NumPy FFN verification successful!")

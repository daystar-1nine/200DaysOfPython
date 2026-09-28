"""
Challenge 4: Implement LayerNorm using NumPy.
"""
import numpy as np


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


if __name__ == "__main__":
    ln = LayerNorm(32)
    x = np.random.randn(2, 5, 32) * 15.0 + 8.0
    out = ln.forward(x)
    print("LN mean:", np.mean(out, axis=-1)[0, :2])
    print("LN var:", np.var(out, axis=-1)[0, :2])
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    assert np.allclose(np.var(out, axis=-1), 1.0, atol=1e-3)
    print("Challenge 4 passed!")

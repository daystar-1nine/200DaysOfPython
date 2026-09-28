"""
Challenge 3: Implement SelfAttention with trainable WQ, WK, WV.
"""
import numpy as np
from challenge_1 import softmax

class SelfAttention:
    def __init__(self, input_dim: int, attention_dim: int, seed: int = 42):
        np.random.seed(seed)
        limit = np.sqrt(6.0 / (input_dim + attention_dim))
        self.Wq = np.random.uniform(-limit, limit, (input_dim, attention_dim))
        self.Wk = np.random.uniform(-limit, limit, (input_dim, attention_dim))
        self.Wv = np.random.uniform(-limit, limit, (input_dim, attention_dim))
        self.d_k = attention_dim

    def forward(self, X: np.ndarray):
        """
        X: (seq_len, input_dim)
        """
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        
        scores = (Q @ K.T) / np.sqrt(self.d_k)
        weights = softmax(scores, axis=-1)
        context = weights @ V
        return context, weights

if __name__ == "__main__":
    X = np.random.randn(5, 10)
    sa = SelfAttention(input_dim=10, attention_dim=16)
    ctx, wts = sa.forward(X)
    print("Context shape:", ctx.shape)
    print("Weights shape:", wts.shape)
    assert ctx.shape == (5, 16)
    assert wts.shape == (5, 5)
    assert np.allclose(wts.sum(axis=-1), 1.0)
    print("Challenge 3 passed!")

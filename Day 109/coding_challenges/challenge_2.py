"""
Challenge 2: Implement scaled_dot_product_attention() using NumPy.
"""
import numpy as np
from challenge_1 import softmax

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """
    Computes Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

if __name__ == "__main__":
    np.random.seed(42)
    Q = np.random.randn(2, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 16)
    
    out, wts = scaled_dot_product_attention(Q, K, V)
    print("Output shape:", out.shape)
    print("Weights shape:", wts.shape)
    assert out.shape == (2, 16)
    assert wts.shape == (2, 4)
    assert np.allclose(wts.sum(axis=-1), 1.0)
    print("Challenge 2 passed!")

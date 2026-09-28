"""
Challenge 4: Implement attention masking.
"""
import numpy as np
from challenge_1 import softmax

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray):
    """
    mask: bool array where False indicates positions that should NOT be attended to.
    """
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    # Apply large negative constant to masked out entries
    scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

if __name__ == "__main__":
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) # 3rd position is pad
    V = np.array([[5.0], [10.0], [0.0]])
    mask = np.array([[True, True, False]])
    
    out, wts = masked_attention(Q, K, V, mask)
    print("Masked weights:", wts)
    assert np.isclose(wts[0, 2], 0.0, atol=1e-5)
    assert np.isclose(wts.sum(), 1.0)
    print("Challenge 4 passed!")

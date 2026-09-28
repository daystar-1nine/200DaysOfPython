"""
Dot-Product Attention and Scaled Dot-Product Attention from scratch in NumPy.
"""
import numpy as np
from softmax_numpy import softmax

def dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """
    Computes unscaled dot-product attention:
      scores = Q @ K.T
      weights = softmax(scores)
      output = weights @ V
    """
    scores = Q @ K.T
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None
):
    """
    Computes scaled dot-product attention:
      scores = (Q @ K.T) / sqrt(d_k)
      if mask is provided:
        scores[mask == 0] = -1e9
      weights = softmax(scores)
      output = weights @ V
      
    Dimensions:
      Q: (..., seq_len_q, d_k)
      K: (..., seq_len_k, d_k)
      V: (..., seq_len_k, d_v)
      mask: broadcastable to (..., seq_len_q, seq_len_k)
    """
    d_k = K.shape[-1]
    # Matrix multiplication along last two dims
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    
    if mask is not None:
        # Where mask is 0 (or False), apply large negative constant
        scores = np.where(mask, scores, -1e9)
        
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

def verify_properties():
    np.random.seed(42)
    seq_len_q = 3
    seq_len_k = 5
    d_k = 8
    d_v = 12
    
    Q = np.random.randn(seq_len_q, d_k)
    K = np.random.randn(seq_len_k, d_k)
    V = np.random.randn(seq_len_k, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("=== Verification of Attention Properties ===")
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output shape:  {output.shape}")
    
    # Property 1: Attention weights are between 0 and 1
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0), "Property 1 failed: Weights not in [0, 1]"
    print("Property 1 Passed: 0 <= weights <= 1")
    
    # Property 2: Each row sums to approximately 1.0
    row_sums = weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), "Property 2 failed: Rows do not sum to 1.0"
    print("Property 2 Passed: Row sums approximately 1.0")
    
    # Property 3: Output shape equals (seq_len_q, d_v)
    assert output.shape == (seq_len_q, d_v), f"Property 3 failed: Expected {(seq_len_q, d_v)}, got {output.shape}"
    print(f"Property 3 Passed: Output shape equals (query_length={seq_len_q}, value_dim={d_v})")
    
    # Masking test
    mask = np.ones((seq_len_q, seq_len_k), dtype=bool)
    mask[:, -2:] = False # Mask out last 2 key positions
    _, masked_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("Masked weights for last 2 keys (should be ~0):", masked_weights[:, -2:])
    assert np.allclose(masked_weights[:, -2:], 0.0, atol=1e-5), "Masking failed"
    print("Masking Verification Passed!")

if __name__ == "__main__":
    verify_properties()

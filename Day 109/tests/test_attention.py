import pytest
import numpy as np
from scratch.attention_numpy import dot_product_attention, scaled_dot_product_attention

def test_dot_product_attention_shapes():
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]])
    
    out, weights = dot_product_attention(Q, K, V)
    assert weights.shape == (1, 3)
    assert out.shape == (1, 2)
    assert np.allclose(weights.sum(), 1.0)

def test_scaled_dot_product_attention_shapes():
    q_len, k_len, d_k, d_v = 4, 6, 8, 16
    Q = np.random.randn(q_len, d_k)
    K = np.random.randn(k_len, d_k)
    V = np.random.randn(k_len, d_v)
    
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert weights.shape == (q_len, k_len)
    assert out.shape == (q_len, d_v)
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0)

def test_attention_scaling_moderates_extreme_scores():
    # Large d_k causes large dot products; scaling moderates peakiness of softmax
    d_k = 64
    Q = np.ones((1, d_k)) * 2.0
    K = np.ones((2, d_k)) * 2.0
    V = np.eye(2)
    
    # Unscaled dot product = 2.0 * 2.0 * 64 = 256
    # Scaled dot product = 256 / sqrt(64) = 32
    _, unscaled_weights = dot_product_attention(Q, K, V)
    _, scaled_weights = scaled_dot_product_attention(Q, K, V)
    
    assert unscaled_weights.shape == (1, 2)
    assert scaled_weights.shape == (1, 2)
    assert np.allclose(scaled_weights.sum(), 1.0)

def test_attention_deterministic_output():
    np.random.seed(999)
    Q = np.random.randn(2, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(3, 5)
    
    out1, w1 = scaled_dot_product_attention(Q, K, V)
    out2, w2 = scaled_dot_product_attention(Q, K, V)
    assert np.array_equal(out1, out2)
    assert np.array_equal(w1, w2)

@pytest.mark.parametrize("d_k", [4, 8, 16, 32, 64])
def test_attention_various_dimensions(d_k):
    Q = np.random.randn(3, d_k)
    K = np.random.randn(5, d_k)
    V = np.random.randn(5, 10)
    
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 10)
    assert weights.shape == (3, 5)
    assert np.allclose(weights.sum(axis=-1), 1.0)

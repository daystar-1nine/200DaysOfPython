"""
Unit tests for Scaled Dot-Product Attention in NumPy and PyTorch.
"""
import pytest
import numpy as np
import torch
from scratch.multi_head_attention import scaled_dot_product_attention, softmax
from app.models.attention import scaled_dot_product_attention_torch


def test_softmax_properties():
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    s = softmax(x, axis=-1)
    assert s.shape == x.shape
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert (s >= 0.0).all() and (s <= 1.0).all()
    assert np.isfinite(s).all()


def test_scaled_dot_product_attention_shapes():
    q = np.random.randn(2, 4, 10, 16)
    k = np.random.randn(2, 4, 10, 16)
    v = np.random.randn(2, 4, 10, 16)
    context, weights = scaled_dot_product_attention(q, k, v)
    assert context.shape == (2, 4, 10, 16)
    assert weights.shape == (2, 4, 10, 10)


def test_scaled_dot_product_attention_weights_sum_to_one():
    q = np.random.randn(3, 8, 15, 32)
    k = np.random.randn(3, 8, 15, 32)
    v = np.random.randn(3, 8, 15, 32)
    _, weights = scaled_dot_product_attention(q, k, v)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_scaled_dot_product_attention_scaling_effect():
    # Large d_k without scaling would yield extreme logits; scaling moderates them
    d_k = 64
    q = np.ones((1, 1, 2, d_k)) * 2.0
    k = np.ones((1, 1, 2, d_k)) * 2.0
    v = np.random.randn(1, 1, 2, d_k)
    _, weights = scaled_dot_product_attention(q, k, v)
    assert np.allclose(weights, 0.5)


@pytest.mark.parametrize("d_k", [8, 16, 32, 64])
def test_attention_various_head_dimensions(d_k):
    q = np.random.randn(1, 2, 5, d_k)
    k = np.random.randn(1, 2, 5, d_k)
    v = np.random.randn(1, 2, 5, d_k)
    ctx, w = scaled_dot_product_attention(q, k, v)
    assert ctx.shape == (1, 2, 5, d_k)
    assert w.shape == (1, 2, 5, 5)


def test_pytorch_scaled_dot_product_attention():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    ctx, w = scaled_dot_product_attention_torch(q, k, v, dropout_p=0.0, training=False)
    assert ctx.shape == (2, 4, 8, 16)
    assert w.shape == (2, 4, 8, 8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5)

"""
Unit tests for Layer Normalization in NumPy and PyTorch.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from scratch.layer_norm import LayerNorm as LN_NumPy


def test_numpy_layer_norm_output_shape():
    ln = LN_NumPy(d_model=64)
    x = np.random.randn(2, 8, 64)
    out = ln.forward(x)
    assert out.shape == x.shape


def test_numpy_layer_norm_zero_mean_unit_variance():
    ln = LN_NumPy(d_model=128)
    x = np.random.randn(4, 15, 128) * 20.0 + 50.0
    out = ln.forward(x)
    mean = np.mean(out, axis=-1)
    var = np.var(out, axis=-1)
    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.allclose(var, 1.0, atol=1e-3)


def test_numpy_layer_norm_numerical_stability():
    ln = LN_NumPy(d_model=32, eps=1e-5)
    # Identical constant inputs along feature dimension
    x = np.ones((2, 5, 32)) * 3.1415
    out = ln.forward(x)
    assert np.isfinite(out).all()
    assert not np.isnan(out).any()


def test_numpy_layer_norm_scale_and_shift():
    ln = LN_NumPy(d_model=16)
    ln.gamma = np.full(16, 2.0, dtype=np.float32)
    ln.beta = np.full(16, 5.0, dtype=np.float32)
    x = np.random.randn(1, 4, 16)
    out = ln.forward(x)
    mean = np.mean(out, axis=-1)
    var = np.var(out, axis=-1)
    assert np.allclose(mean, 5.0, atol=1e-5)
    assert np.allclose(var, 4.0, atol=1e-3)


def test_numpy_vs_pytorch_layer_norm_parity():
    d_model = 32
    ln_np = LN_NumPy(d_model=d_model, eps=1e-5)
    ln_pt = nn.LayerNorm(d_model, eps=1e-5)

    x = np.random.randn(2, 6, d_model).astype(np.float32)
    out_np = ln_np.forward(x)
    with torch.no_grad():
        out_pt = ln_pt(torch.from_numpy(x)).numpy()

    assert np.allclose(out_np, out_pt, atol=1e-5)


def test_layer_norm_4d_tensor_support():
    ln = LN_NumPy(d_model=16)
    x = np.random.randn(2, 3, 4, 16) * 5.0 + 3.0
    out = ln.forward(x)
    assert out.shape == (2, 3, 4, 16)
    mean = np.mean(out, axis=-1)
    var = np.var(out, axis=-1)
    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.allclose(var, 1.0, atol=1e-3)


def test_pytorch_layer_norm_gradient_flow():
    ln = nn.LayerNorm(32)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = ln(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None

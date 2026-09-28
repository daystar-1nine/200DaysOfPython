"""
Unit tests for Position-wise Feed-Forward Networks.
"""
import pytest
import numpy as np
import torch
from scratch.feed_forward import FeedForward as FFN_NumPy
from app.models.feed_forward import PositionwiseFeedForward as FFN_Torch


def test_numpy_ffn_shape_preservation():
    ffn = FFN_NumPy(d_model=64, d_ff=128)
    x = np.random.randn(3, 12, 64)
    out = ffn.forward(x)
    assert out.shape == x.shape


def test_numpy_ffn_nonlinearity():
    ffn = FFN_NumPy(d_model=32, d_ff=64)
    x1 = np.random.randn(1, 5, 32)
    x2 = np.random.randn(1, 5, 32)
    out_sum = ffn.forward(x1 + x2)
    sum_out = ffn.forward(x1) + ffn.forward(x2)
    # Because of non-linear ReLU/GELU activation, f(a+b) != f(a) + f(b)
    assert not np.allclose(out_sum, sum_out)


def test_numpy_ffn_gelu_activation():
    ffn = FFN_NumPy(d_model=32, d_ff=64, activation="gelu")
    x = np.random.randn(2, 4, 32)
    out = ffn.forward(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_pytorch_ffn_shape_and_forward():
    ffn = FFN_Torch(d_model=64, d_ff=256, dropout=0.0)
    x = torch.randn(4, 10, 64)
    out = ffn(x)
    assert out.shape == x.shape


def test_pytorch_ffn_parameter_count():
    d_model = 64
    d_ff = 128
    ffn = FFN_Torch(d_model=d_model, d_ff=d_ff, dropout=0.0)
    # linear1: (64 * 128 + 128), linear2: (128 * 64 + 64)
    expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    actual_params = sum(p.numel() for p in ffn.parameters())
    assert actual_params == expected_params

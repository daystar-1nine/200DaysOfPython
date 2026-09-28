"""
Unit tests for Sinusoidal Positional Encoding (NumPy and PyTorch).
"""
import pytest
import numpy as np
import torch
from scratch.positional_encoding import positional_encoding
from app.preprocessing.positional_encoding import PositionalEncoding, get_positional_encoding_numpy


def test_positional_encoding_correct_shape():
    for max_len, d_model in [(50, 64), (100, 128), (20, 32), (150, 256)]:
        pe = positional_encoding(max_len, d_model)
        assert pe.shape == (max_len, d_model)


def test_positional_encoding_deterministic():
    pe1 = positional_encoding(60, 64)
    pe2 = positional_encoding(60, 64)
    assert np.allclose(pe1, pe2)


def test_positional_encoding_finite_values():
    pe = positional_encoding(100, 128)
    assert np.isfinite(pe).all()
    assert not np.isnan(pe).any()


def test_positional_encoding_position_zero():
    pe = positional_encoding(10, 32)
    # Even dimensions at position 0 are sin(0) = 0
    assert np.allclose(pe[0, 0::2], 0.0, atol=1e-5)
    # Odd dimensions at position 0 are cos(0) = 1
    assert np.allclose(pe[0, 1::2], 1.0, atol=1e-5)


def test_positional_encoding_different_positions_differ():
    pe = positional_encoding(50, 64)
    for pos in range(1, 10):
        assert not np.allclose(pe[0], pe[pos])
        assert not np.allclose(pe[pos], pe[pos + 1])


def test_positional_encoding_different_dimensions_differ():
    pe = positional_encoding(50, 64)
    for d in range(0, 10):
        assert not np.allclose(pe[:, d], pe[:, d + 1])


def test_positional_encoding_invalid_parameters():
    with pytest.raises(ValueError):
        positional_encoding(-1, 128)
    with pytest.raises(ValueError):
        positional_encoding(100, 0)


def test_pytorch_positional_encoding_shape():
    pe_layer = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
    x = torch.randn(4, 25, 64)
    out = pe_layer(x)
    assert out.shape == (4, 25, 64)


def test_pytorch_positional_encoding_registered_buffer():
    pe_layer = PositionalEncoding(d_model=32, max_len=50)
    buffers = dict(pe_layer.named_buffers())
    assert "pe" in buffers
    assert buffers["pe"].shape == (1, 50, 32)


def test_positional_encoding_frequency_decay():
    # Lower dimension indices correspond to higher frequency signals
    pe = positional_encoding(100, 64)
    # Check that high frequencies (dim 0) alternate signs faster than low frequencies (dim 62)
    zero_crossings_dim0 = np.sum(np.diff(np.sign(pe[:, 0])) != 0)
    zero_crossings_dim62 = np.sum(np.diff(np.sign(pe[:, 62])) != 0)
    assert zero_crossings_dim0 > zero_crossings_dim62


def test_positional_encoding_dot_product_distance_monotonicity():
    # Dot product of PE(pos) with PE(pos + k) decays as k increases
    pe = positional_encoding(50, 64)
    dot_k1 = np.dot(pe[10], pe[11])
    dot_k10 = np.dot(pe[10], pe[20])
    dot_k30 = np.dot(pe[10], pe[40])
    assert dot_k1 > dot_k10
    assert dot_k1 > dot_k30


def test_pytorch_positional_encoding_batch_invariance():
    pe_layer = PositionalEncoding(d_model=32, max_len=50, dropout=0.0)
    x = torch.zeros(3, 10, 32)
    out = pe_layer(x)
    # All batch elements should have received the exact same positional vectors
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[1], out[2])

"""
Unit tests for Multi-Head Attention mechanisms in NumPy and PyTorch.
"""
import pytest
import numpy as np
import torch
from scratch.multi_head_attention import MultiHeadAttention as MHA_NumPy
from app.models.multi_head_attention import MultiHeadAttention as MHA_Torch


def test_numpy_mha_invalid_dimensions():
    with pytest.raises(ValueError):
        MHA_NumPy(d_model=100, num_heads=3)  # 100 not divisible by 3


def test_pytorch_mha_invalid_dimensions():
    with pytest.raises(ValueError):
        MHA_Torch(d_model=100, num_heads=3)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_numpy_mha_head_counts(num_heads):
    d_model = 64
    mha = MHA_NumPy(d_model=d_model, num_heads=num_heads)
    x = np.random.randn(2, 6, d_model)
    out, weights = mha.forward(x, x, x)
    assert out.shape == (2, 6, d_model)
    assert weights.shape == (2, num_heads, 6, 6)


@pytest.mark.parametrize("num_heads", [2, 4, 8])
def test_pytorch_mha_head_counts(num_heads):
    d_model = 64
    mha = MHA_Torch(d_model=d_model, num_heads=num_heads, dropout=0.0)
    x = torch.randn(2, 6, d_model)
    out, weights = mha(x, x, x)
    assert out.shape == (2, 6, d_model)
    assert weights.shape == (2, num_heads, 6, 6)


def test_numpy_mha_split_and_combine_identity():
    d_model = 32
    num_heads = 4
    mha = MHA_NumPy(d_model=d_model, num_heads=num_heads)
    x = np.random.randn(3, 5, d_model)
    split = mha._split_heads(x)
    assert split.shape == (3, num_heads, 5, d_model // num_heads)
    recombined = mha._combine_heads(split)
    assert np.allclose(x, recombined)


def test_pytorch_mha_split_and_combine_identity():
    d_model = 32
    num_heads = 4
    mha = MHA_Torch(d_model=d_model, num_heads=num_heads)
    x = torch.randn(3, 5, d_model)
    split = mha._split_heads(x)
    assert split.shape == (3, num_heads, 5, d_model // num_heads)
    recombined = mha._combine_heads(split)
    assert torch.allclose(x, recombined)


def test_mha_different_query_and_key_lengths():
    d_model = 32
    num_heads = 2
    mha = MHA_Torch(d_model=d_model, num_heads=num_heads, dropout=0.0)
    q = torch.randn(2, 4, d_model)   # seq_len_q = 4
    kv = torch.randn(2, 7, d_model)  # seq_len_k = 7
    out, weights = mha(q, kv, kv)
    assert out.shape == (2, 4, d_model)
    assert weights.shape == (2, num_heads, 4, 7)


def test_mha_projection_weights_shapes():
    d_model = 64
    num_heads = 4
    mha_np = MHA_NumPy(d_model=d_model, num_heads=num_heads)
    assert mha_np.w_q.shape == (d_model, d_model)
    assert mha_np.w_k.shape == (d_model, d_model)
    assert mha_np.w_v.shape == (d_model, d_model)
    assert mha_np.w_o.shape == (d_model, d_model)


def test_self_attention_permutation_equivariance_without_pe():
    # In pure self-attention without positional encoding, permuting input tokens
    # permutes the output tokens in the exact same permutation order.
    d_model = 16
    mha = MHA_Torch(d_model=d_model, num_heads=2, dropout=0.0)
    mha.eval()
    x = torch.randn(1, 4, d_model)
    perm = [2, 0, 3, 1]
    x_perm = x[:, perm, :]

    with torch.no_grad():
        out, _ = mha(x, x, x)
        out_perm, _ = mha(x_perm, x_perm, x_perm)

    assert torch.allclose(out[:, perm, :], out_perm, atol=1e-5)

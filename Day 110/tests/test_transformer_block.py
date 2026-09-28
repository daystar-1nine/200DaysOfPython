"""
Unit tests for Transformer Block in NumPy and PyTorch.
"""
import pytest
import numpy as np
import torch
from scratch.transformer_block import TransformerBlock as TB_NumPy
from app.models.transformer_block import TransformerEncoderBlock as TB_Torch


def test_numpy_transformer_block_forward_shapes():
    block = TB_NumPy(d_model=64, num_heads=4, d_ff=128)
    x = np.random.randn(2, 8, 64)
    out, weights = block.forward(x)
    assert out.shape == (2, 8, 64)
    assert weights.shape == (2, 4, 8, 8)
    assert np.isfinite(out).all()


def test_numpy_transformer_block_with_padding_mask():
    block = TB_NumPy(d_model=32, num_heads=2, d_ff=64)
    x = np.random.randn(2, 6, 32)
    # Mask out last 2 tokens
    mask = np.zeros((2, 6), dtype=bool)
    mask[:, 4:] = True
    out, weights = block.forward(x, mask=mask)
    assert out.shape == (2, 6, 32)
    # Masked positions should have zero attention weight
    assert np.allclose(weights[:, :, :, 4:], 0.0, atol=1e-5)


def test_pytorch_transformer_block_forward():
    block = TB_Torch(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
    block.eval()
    x = torch.randn(3, 10, 64)
    out, weights = block(x)
    assert out.shape == (3, 10, 64)
    assert weights.shape == (3, 4, 10, 10)


def test_pytorch_transformer_block_ablation_flags():
    # Without residual
    b_no_res = TB_Torch(d_model=32, num_heads=2, d_ff=64, use_residual=False)
    # Without layernorm
    b_no_ln = TB_Torch(d_model=32, num_heads=2, d_ff=64, use_layer_norm=False)
    x = torch.randn(2, 5, 32)
    o1, _ = b_no_res(x)
    o2, _ = b_no_ln(x)
    assert o1.shape == x.shape
    assert o2.shape == x.shape


def test_transformer_block_gradient_flow():
    block = TB_Torch(d_model=32, num_heads=2, d_ff=64)
    x = torch.randn(2, 4, 32, requires_grad=True)
    out, _ = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    for name, p in block.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"


def test_transformer_block_eval_mode_deterministic():
    block = TB_Torch(d_model=32, num_heads=2, d_ff=64, dropout=0.5)
    block.eval()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        out1, _ = block(x)
        out2, _ = block(x)
    assert torch.allclose(out1, out2)

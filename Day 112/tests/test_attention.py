"""
Unit tests for Causal Self-Attention mechanisms in PyTorch and NumPy.
"""
import pytest
import numpy as np
import torch
from app.model.attention import CausalSelfAttention
from scratch.attention import (
    CausalSelfAttentionScratch,
    scaled_dot_product_causal_attention_np
)


class TestCausalSelfAttention:
    def test_output_shape_matches_input(self):
        B, T, C = 2, 8, 32
        x = torch.randn(B, T, C)
        attn = CausalSelfAttention(embed_dim=C, num_heads=4, context_length=16)
        out, _ = attn(x)
        assert out.shape == (B, T, C)

    def test_output_contains_no_nans(self):
        x = torch.randn(3, 10, 64)
        attn = CausalSelfAttention(embed_dim=64, num_heads=4, context_length=16)
        out, _ = attn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_embed_dim_not_divisible_by_num_heads_raises_error(self):
        with pytest.raises(ValueError):
            CausalSelfAttention(embed_dim=33, num_heads=4, context_length=16)

    def test_attention_weights_properties(self):
        B, T, C, H = 2, 6, 32, 4
        x = torch.randn(B, T, C)
        attn = CausalSelfAttention(embed_dim=C, num_heads=H, context_length=16)
        attn.eval()
        _, weights = attn(x, return_weights=True)

        assert weights.shape == (B, H, T, T)
        # 1. Weights must sum to 1.0 along the key dimension
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        # 2. Causal property: upper triangle strictly zero
        for b in range(B):
            for h in range(H):
                upper = torch.triu(weights[b, h], diagonal=1)
                assert torch.all(upper == 0.0)

    def test_backprop_gradients_flow(self):
        x = torch.randn(2, 4, 32, requires_grad=True)
        attn = CausalSelfAttention(embed_dim=32, num_heads=2, context_length=8)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.c_attn.weight.grad is not None
        assert attn.c_proj.weight.grad is not None

    def test_causal_invariance_future_tokens_do_not_affect_past(self):
        """
        Fundamental property of causal masking:
        Modifying future tokens at t > k must NOT change the representations at t <= k.
        """
        B, T, C = 1, 6, 32
        attn = CausalSelfAttention(embed_dim=C, num_heads=4, context_length=16, dropout=0.0)
        attn.eval()

        # Sequence 1
        x1 = torch.randn(B, T, C)
        # Sequence 2: identical up to index 3, but completely different at indices 4 and 5
        x2 = x1.clone()
        x2[:, 4:, :] = torch.randn(B, 2, C) * 10.0

        with torch.no_grad():
            out1, _ = attn(x1)
            out2, _ = attn(x2)

        # Positions 0, 1, 2, 3 must be numerically identical
        assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)
        # Positions 4 and 5 must differ because future tokens were changed
        assert not torch.allclose(out1[:, 4:, :], out2[:, 4:, :], atol=1e-3)

    def test_single_token_sequence_boundary(self):
        x = torch.randn(1, 1, 32)
        attn = CausalSelfAttention(embed_dim=32, num_heads=2, context_length=8)
        attn.eval()
        out, weights = attn(x, return_weights=True)
        assert out.shape == (1, 1, 32)
        assert weights.shape == (1, 2, 1, 1)
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_eval_mode_deterministic_outputs(self):
        x = torch.randn(2, 4, 32)
        attn = CausalSelfAttention(embed_dim=32, num_heads=4, context_length=8, dropout=0.5)
        attn.eval()
        with torch.no_grad():
            out1, _ = attn(x)
            out2, _ = attn(x)
        assert torch.equal(out1, out2)


class TestNumPyAttentionScratch:
    def test_numpy_scaled_dot_product_causal(self):
        T, D = 4, 8
        q = np.random.randn(T, D)
        k = np.random.randn(T, D)
        v = np.random.randn(T, D)
        mask = np.tril(np.ones((T, T)))

        out, weights = scaled_dot_product_causal_attention_np(q, k, v, mask=mask)
        assert out.shape == (T, D)
        assert weights.shape == (T, T)
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(np.triu(weights, k=1) == 0.0)

    def test_scratch_multi_head_shape_and_no_nan(self):
        scratch_attn = CausalSelfAttentionScratch(embed_dim=32, num_heads=4, context_length=16)
        x = torch.randn(2, 6, 32)
        out, weights = scratch_attn.forward(x, return_weights=True)
        assert out.shape == (2, 6, 32)
        assert weights.shape == (2, 4, 6, 6)
        assert not torch.isnan(out).any()
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-5)

"""
Unit tests for Causal Mask implementations across NumPy and PyTorch.
"""
import pytest
import numpy as np
import torch
import torch.nn.functional as F
from scratch.causal_mask import (
    causal_mask_numpy,
    causal_mask_torch,
    create_additive_causal_mask
)


class TestCausalMaskNumPy:
    def test_shape_matches_sequence_length(self):
        for seq_len in [1, 5, 16, 64]:
            mask = causal_mask_numpy(seq_len)
            assert mask.shape == (seq_len, seq_len)

    def test_diagonal_and_lower_triangle_ones(self):
        mask = causal_mask_numpy(8)
        # All elements on and below diagonal must be 1
        for i in range(8):
            for j in range(8):
                if j <= i:
                    assert mask[i, j] == 1
                else:
                    assert mask[i, j] == 0

    def test_upper_triangle_strictly_zeros(self):
        mask = causal_mask_numpy(12)
        upper = np.triu(mask, k=1)
        assert np.all(upper == 0)

    def test_numpy_mask_dtype(self):
        mask = causal_mask_numpy(4)
        assert mask.dtype in (np.int32, np.int64, int)


class TestCausalMaskTorch:
    def test_torch_boolean_mask(self):
        mask = causal_mask_torch(10)
        assert mask.dtype == torch.bool
        assert mask.shape == (10, 10)
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is False
        assert mask[9, 0].item() is True

    def test_additive_mask_values(self):
        mask = create_additive_causal_mask(6)
        assert mask.shape == (6, 6)
        # Allowed positions must be 0.0
        assert mask[0, 0].item() == 0.0
        assert mask[5, 5].item() == 0.0
        assert mask[5, 0].item() == 0.0
        # Blocked positions must be -inf
        assert torch.isneginf(mask[0, 1])
        assert torch.isneginf(mask[0, 5])
        assert torch.isneginf(mask[4, 5])

    def test_single_token_sequence_boundary(self):
        mask = create_additive_causal_mask(1)
        assert mask.shape == (1, 1)
        assert mask[0, 0].item() == 0.0

    def test_broadcasting_compatibility(self):
        mask = create_additive_causal_mask(8)
        # Verify it can broadcast with (batch_size, num_heads, seq_len, seq_len)
        scores = torch.randn(2, 4, 8, 8)
        masked_scores = scores + mask
        assert masked_scores.shape == (2, 4, 8, 8)
        # Check that future positions in masked_scores are -inf
        assert torch.all(torch.isneginf(masked_scores[:, :, 0, 1]))

    def test_causal_mask_softmax_probabilities(self):
        """Applying softmax to masked scores produces strictly zero probability for future tokens."""
        T = 5
        mask = create_additive_causal_mask(T)
        scores = torch.randn(1, 1, T, T)
        weights = F.softmax(scores + mask, dim=-1)

        # Upper triangle probabilities must be exactly 0.0
        upper = torch.triu(weights[0, 0], diagonal=1)
        assert torch.all(upper == 0.0)

        # Lower triangle + diagonal must sum to 1.0 along the row
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, T), atol=1e-5)

    def test_additive_mask_device_and_dtype(self):
        mask = create_additive_causal_mask(4, device=torch.device("cpu"))
        assert mask.device.type == "cpu"
        assert mask.dtype == torch.float32

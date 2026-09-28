"""
Unit tests for Padding and Causal Masks.
"""
import pytest
import numpy as np
import torch
from app.preprocessing.padding import create_padding_mask, create_padding_mask_torch, causal_mask
from scratch.multi_head_attention import scaled_dot_product_attention


def test_create_padding_mask_numpy():
    token_ids = np.array([
        [12, 45, 9, 0, 0],
        [3, 88, 12, 4, 0]
    ])
    mask = create_padding_mask(token_ids, pad_id=0)
    assert mask.shape == (2, 5)
    assert mask.dtype == bool
    assert np.array_equal(mask[0], [False, False, False, True, True])
    assert np.array_equal(mask[1], [False, False, False, False, True])


def test_create_padding_mask_torch():
    token_ids = torch.tensor([
        [10, 20, 0, 0],
        [5, 6, 7, 0]
    ])
    mask = create_padding_mask_torch(token_ids, pad_id=0)
    assert mask.shape == (2, 4)
    assert mask[0, 2].item() is True
    assert mask[0, 0].item() is False


def test_causal_mask_properties():
    length = 5
    mask = causal_mask(length)
    assert mask.shape == (5, 5)
    assert mask.dtype == bool

    # Main diagonal must be False (each token attends to itself)
    for i in range(length):
        assert not mask[i, i]

    # Lower triangular elements must be False (can attend to past)
    for i in range(length):
        for j in range(i):
            assert not mask[i, j]

    # Upper triangular strictly above diagonal must be True (cannot attend to future)
    for i in range(length):
        for j in range(i + 1, length):
            assert mask[i, j]


def test_attention_mask_zeros_out_attention_weight():
    q = np.random.randn(1, 1, 3, 8)
    k = np.random.randn(1, 1, 3, 8)
    v = np.random.randn(1, 1, 3, 8)

    # Mask the third key token
    mask = np.array([[[[False, False, True]]]])
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    assert np.allclose(weights[:, :, :, 2], 0.0, atol=1e-5)
    # The remaining two positions must sum to 1.0
    assert np.allclose(weights[:, :, :, :2].sum(axis=-1), 1.0, atol=1e-5)

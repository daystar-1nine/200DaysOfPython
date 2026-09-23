"""
Unit tests for masked global average pooling.
Day 105: Neural NLP & Text Classification.
"""

import numpy as np
import pytest
import torch
from app.embeddings.pooling import (
    masked_global_average_pooling_np,
    masked_global_average_pooling
)


def test_masked_pooling_single_sequence():
    # 3 tokens, dim 2
    embs = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [999.0, 999.0]  # <PAD>
    ], dtype=np.float32)
    mask = np.array([1, 1, 0], dtype=np.float32)

    pooled = masked_global_average_pooling_np(embs, mask)
    expected = np.array([2.0, 3.0], dtype=np.float32)
    assert np.allclose(pooled, expected)


def test_masked_pooling_batch():
    # Batch size 2, seq len 3, dim 2
    embs = np.array([
        [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
        [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0]]
    ], dtype=np.float32)
    mask = np.array([
        [1, 1, 0],
        [1, 0, 0]
    ], dtype=np.float32)

    pooled = masked_global_average_pooling_np(embs, mask)
    assert pooled.shape == (2, 2)
    assert np.allclose(pooled[0], [1.5, 1.5])
    assert np.allclose(pooled[1], [3.0, 3.0])


def test_masked_pooling_all_padding():
    embs = np.array([
        [[5.0, 5.0], [6.0, 6.0]]
    ], dtype=np.float32)
    mask = np.array([[0, 0]], dtype=np.float32)
    pooled = masked_global_average_pooling_np(embs, mask)
    assert np.allclose(pooled, np.zeros((1, 2)))


def test_masked_pooling_torch_matches_numpy():
    embs_np = np.random.randn(4, 5, 8).astype(np.float32)
    mask_np = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0]
    ], dtype=np.float32)

    pooled_np = masked_global_average_pooling_np(embs_np, mask_np)

    embs_torch = torch.tensor(embs_np)
    mask_torch = torch.tensor(mask_np)
    pooled_torch = masked_global_average_pooling(embs_torch, mask_torch).numpy()

    assert np.allclose(pooled_np, pooled_torch, atol=1e-5)

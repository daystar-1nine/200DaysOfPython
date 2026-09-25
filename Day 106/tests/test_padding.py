"""
Tests for SequencePadder, truncation, and masking in Day 106.
"""

import numpy as np
import pytest
from app.preprocessing.padding import SequencePadder


def test_padding_shorter_sequence_post():
    seq = [2, 3, 4]
    padded = SequencePadder.pad_sequence(seq, max_length=6, padding="post", pad_value=0)
    assert padded == [2, 3, 4, 0, 0, 0]


def test_padding_shorter_sequence_pre():
    seq = [2, 3, 4]
    padded = SequencePadder.pad_sequence(seq, max_length=6, padding="pre", pad_value=0)
    assert padded == [0, 0, 0, 2, 3, 4]


def test_truncation_longer_sequence_post():
    seq = [1, 2, 3, 4, 5, 6, 7]
    truncated = SequencePadder.pad_sequence(seq, max_length=4, truncating="post")
    assert truncated == [1, 2, 3, 4]


def test_truncation_longer_sequence_pre():
    seq = [1, 2, 3, 4, 5, 6, 7]
    truncated = SequencePadder.pad_sequence(seq, max_length=4, truncating="pre")
    assert truncated == [4, 5, 6, 7]


def test_pad_batch_shapes_and_types():
    batch = [[2, 3], [4, 5, 6, 7], [8]]
    padded, mask = SequencePadder.pad_batch(batch, max_length=5, pad_value=0)
    assert padded.shape == (3, 5)
    assert mask.shape == (3, 5)
    assert padded.dtype == np.int64
    assert mask.dtype == np.float32


def test_mask_generation_accuracy():
    batch = [[2, 3], [4, 5, 6]]
    padded, mask = SequencePadder.pad_batch(batch, max_length=4, padding="post", pad_value=0)
    # Row 0: [2, 3, 0, 0] -> mask: [1, 1, 0, 0]
    np.testing.assert_array_equal(mask[0], [1.0, 1.0, 0.0, 0.0])
    # Row 1: [4, 5, 6, 0] -> mask: [1, 1, 1, 0]
    np.testing.assert_array_equal(mask[1], [1.0, 1.0, 1.0, 0.0])

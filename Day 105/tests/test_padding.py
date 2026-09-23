"""
Unit tests for sequence padding and masking.
Day 105: Neural NLP & Text Classification.
"""

import numpy as np
import pytest
from app.preprocessing.padding import pad_sequence, create_mask, pad_batch


def test_pad_sequence_shorter():
    seq = [1, 2, 3]
    padded, mask = pad_sequence(seq, max_length=5, pad_id=0)
    assert padded == [1, 2, 3, 0, 0]
    assert mask == [1, 1, 1, 0, 0]


def test_pad_sequence_exact():
    seq = [1, 2, 3, 4]
    padded, mask = pad_sequence(seq, max_length=4, pad_id=0)
    assert padded == [1, 2, 3, 4]
    assert mask == [1, 1, 1, 1]


def test_pad_sequence_longer_post_truncation():
    seq = [1, 2, 3, 4, 5, 6]
    padded, mask = pad_sequence(seq, max_length=3, truncation_strategy="post")
    assert padded == [1, 2, 3]
    assert mask == [1, 1, 1]


def test_pad_sequence_longer_pre_truncation():
    seq = [1, 2, 3, 4, 5, 6]
    padded, mask = pad_sequence(seq, max_length=3, truncation_strategy="pre")
    assert padded == [4, 5, 6]
    assert mask == [1, 1, 1]


def test_pad_sequence_pre_padding():
    seq = [1, 2]
    padded, mask = pad_sequence(seq, max_length=4, padding_strategy="pre", pad_id=0)
    assert padded == [0, 0, 1, 2]
    assert mask == [0, 0, 1, 1]


def test_pad_sequence_empty():
    padded, mask = pad_sequence([], max_length=3, pad_id=0)
    assert padded == [0, 0, 0]
    assert mask == [0, 0, 0]


def test_create_mask():
    seq = [5, 6, 7, 0, 0]
    mask = create_mask(seq, pad_id=0)
    assert mask == [1, 1, 1, 0, 0]


def test_pad_batch():
    batch = [[1, 2], [3, 4, 5, 6], [7]]
    padded, masks = pad_batch(batch, max_length=4, pad_id=0)
    assert padded.shape == (3, 4)
    assert masks.shape == (3, 4)
    assert np.allclose(padded[0], [1, 2, 0, 0])
    assert np.allclose(masks[0], [1, 1, 0, 0])
    assert np.allclose(padded[1], [3, 4, 5, 6])
    assert np.allclose(masks[1], [1, 1, 1, 1])

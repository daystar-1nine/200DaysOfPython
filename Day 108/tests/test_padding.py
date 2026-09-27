import pytest
import numpy as np
from app.preprocessing.padding import pad_sequence, pad_sequences
from app.config import PAD_IDX

def test_pad_sequence_padding_post():
    seq = [10, 20]
    padded = pad_sequence(seq, max_len=5, pad_value=0, padding="post")
    assert padded == [10, 20, 0, 0, 0]

def test_pad_sequence_padding_pre():
    seq = [10, 20]
    padded = pad_sequence(seq, max_len=5, pad_value=0, padding="pre")
    assert padded == [0, 0, 0, 10, 20]

def test_pad_sequence_truncation_post():
    seq = [1, 2, 3, 4, 5, 6]
    truncated = pad_sequence(seq, max_len=4, truncating="post")
    assert truncated == [1, 2, 3, 4]

def test_pad_sequence_truncation_pre():
    seq = [1, 2, 3, 4, 5, 6]
    truncated = pad_sequence(seq, max_len=4, truncating="pre")
    assert truncated == [3, 4, 5, 6]

def test_pad_sequences_batch_shape():
    batch = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9, 10]
    ]
    arr = pad_sequences(batch, max_len=4, pad_value=PAD_IDX)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 4)
    assert list(arr[0]) == [1, 2, 3, 0]
    assert list(arr[1]) == [4, 5, 0, 0]
    assert list(arr[2]) == [6, 7, 8, 9]

@pytest.mark.parametrize("target_len", [5, 10, 20, 50])
def test_pad_sequences_various_lengths(target_len):
    batch = [[1, 2, 3], [4]]
    arr = pad_sequences(batch, max_len=target_len)
    assert arr.shape == (2, target_len)

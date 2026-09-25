"""
Tests for pure NumPy scratch RNN implementation in Day 106.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure scratch directory is discoverable
SCRATCH_DIR = Path(__file__).resolve().parent.parent / "scratch"
sys.path.insert(0, str(SCRATCH_DIR))

from simple_rnn import rnn_step, forward_sequence
from rnn_shapes import calculate_rnn_parameters


def test_scratch_rnn_step_shape():
    x_t = np.random.randn(8)
    h_prev = np.random.randn(16)
    Wx = np.random.randn(16, 8)
    Wh = np.random.randn(16, 16)
    b = np.zeros(16)

    h_new = rnn_step(x_t, h_prev, Wx, Wh, b)
    assert h_new.shape == (16,)
    # tanh bounds output in (-1, 1)
    assert np.all(h_new >= -1.0) and np.all(h_new <= 1.0)


def test_scratch_forward_sequence_shapes():
    seq = np.random.randn(7, 10).astype(np.float32)
    Wx = np.random.randn(20, 10).astype(np.float32)
    Wh = np.random.randn(20, 20).astype(np.float32)
    b = np.zeros(20, dtype=np.float32)

    all_states, final_state = forward_sequence(seq, Wx, Wh, b)
    assert all_states.shape == (7, 20)
    assert final_state.shape == (20,)
    np.testing.assert_allclose(all_states[-1], final_state)


def test_scratch_rnn_deterministic():
    seq = np.ones((4, 5), dtype=np.float32)
    Wx = np.ones((8, 5), dtype=np.float32) * 0.1
    Wh = np.ones((8, 8), dtype=np.float32) * 0.1
    b = np.zeros(8, dtype=np.float32)

    s1, f1 = forward_sequence(seq, Wx, Wh, b)
    s2, f2 = forward_sequence(seq, Wx, Wh, b)
    np.testing.assert_allclose(s1, s2)
    np.testing.assert_allclose(f1, f2)


def test_scratch_order_sensitivity():
    # Inverted tokens produce different final hidden states
    t1 = np.array([1.0, 0.0], dtype=np.float32)
    t2 = np.array([0.0, 1.0], dtype=np.float32)

    seq_a = np.array([t1, t2])
    seq_b = np.array([t2, t1])

    Wx = np.array([[0.5, 0.2], [-0.3, 0.4]], dtype=np.float32)
    Wh = np.array([[0.1, 0.3], [0.2, -0.1]], dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)

    _, f_a = forward_sequence(seq_a, Wx, Wh, b)
    _, f_b = forward_sequence(seq_b, Wx, Wh, b)
    assert not np.allclose(f_a, f_b)


def test_calculate_rnn_parameters_counts():
    counts = calculate_rnn_parameters(input_dim=32, hidden_size=64, output_dim=1)
    # Wx: 64*32=2048, Wh: 64*64=4096, b: 64 -> RNN total: 6208
    assert counts["Wx_params"] == 2048
    assert counts["Wh_params"] == 4096
    assert counts["b_params"] == 64
    assert counts["rnn_total"] == 6208
    assert counts["dense_total"] == 65
    assert counts["grand_total"] == 6273

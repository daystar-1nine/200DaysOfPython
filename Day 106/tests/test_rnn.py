"""
Tests for PyTorch SimpleRNNClassifier architecture and forward pass in Day 106.
"""

import numpy as np
import pytest
import torch
from app.models.rnn import SimpleRNNClassifier


def test_rnn_classifier_forward_shape():
    model = SimpleRNNClassifier(vocab_size=100, embedding_dim=16, hidden_units=32, dense_units=16)
    x = torch.randint(0, 100, (4, 15), dtype=torch.long)
    mask = torch.ones((4, 15), dtype=torch.float32)

    logits = model(x, mask)
    assert logits.shape == (4,)


def test_rnn_return_sequences():
    model = SimpleRNNClassifier(vocab_size=50, embedding_dim=16, hidden_units=24)
    x = torch.randint(0, 50, (3, 10), dtype=torch.long)

    seq_states = model(x, return_sequences=True)
    assert seq_states.shape == (3, 10, 24)


def test_rnn_return_state():
    model = SimpleRNNClassifier(vocab_size=50, embedding_dim=16, hidden_units=24)
    x = torch.randint(0, 50, (3, 10), dtype=torch.long)

    logits, final_state = model(x, return_state=True)
    assert logits.shape == (3,)
    assert final_state.shape == (3, 24)


def test_rnn_predict_proba_range():
    model = SimpleRNNClassifier(vocab_size=50, embedding_dim=16, hidden_units=16)
    x = torch.randint(0, 50, (5, 8), dtype=torch.long)
    mask = torch.ones((5, 8), dtype=torch.float32)

    probs = model.predict_proba(x, mask)
    assert probs.shape == (5,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_rnn_parameter_breakdown():
    # vocab=100, emb=16, hidden=32, dense=16, out=1
    model = SimpleRNNClassifier(vocab_size=100, embedding_dim=16, hidden_units=32, dense_units=16)
    counts = model.count_parameters()
    # emb: 100 * 16 = 1600
    assert counts["embedding"] == 1600
    # rnn: (16 + 32) * 32 + 32 + 32 (PyTorch RNN has input and hidden bias) = 48*32 + 64 = 1600
    assert counts["rnn"] == 1600
    assert counts["total"] > 3000


def test_rnn_padding_idx_zero_grad():
    model = SimpleRNNClassifier(vocab_size=50, embedding_dim=16, hidden_units=16, padding_idx=0)
    x = torch.zeros((2, 5), dtype=torch.long)
    mask = torch.ones((2, 5), dtype=torch.float32)

    logits = model(x, mask)
    loss = logits.sum()
    loss.backward()

    # The gradient for padding token index 0 should be exactly 0
    pad_grad = model.embedding.weight.grad[0]
    assert torch.all(pad_grad == 0.0)

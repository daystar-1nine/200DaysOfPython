"""
Unit tests for neural text classifier architectures.
Day 105: Neural NLP & Text Classification.
"""

import pytest
import torch
from app.models.architectures import build_model_a, build_model_b, build_model_c
from app.models.embedding_classifier import NeuralTextClassifier


def test_model_a_forward():
    model = build_model_a(vocab_size=100, embedding_dim=16)
    x = torch.randint(0, 100, (4, 10))
    mask = torch.ones(4, 10)
    out = model(x, mask=mask)
    assert out.shape == (4,)


def test_model_b_forward():
    model = build_model_b(vocab_size=100, embedding_dim=16, hidden_dim=32)
    x = torch.randint(0, 100, (4, 10))
    mask = torch.ones(4, 10)
    out = model(x, mask=mask)
    assert out.shape == (4,)


def test_model_c_forward():
    model = build_model_c(vocab_size=100, embedding_dim=16, hidden_dim=32, dropout_rate=0.5)
    x = torch.randint(0, 100, (4, 10))
    mask = torch.ones(4, 10)
    out = model(x, mask=mask)
    assert out.shape == (4,)


def test_model_predict_proba():
    model = build_model_c(vocab_size=50, embedding_dim=16, hidden_dim=8)
    x = torch.randint(0, 50, (6, 8))
    probs = model.predict_proba(x)
    assert probs.shape == (6,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_model_parameter_count():
    # V=100, D=16, H=32
    # Embedding: 100 * 16 = 1600
    # FC1: 16 * 32 + 32 = 544
    # FC2: 32 * 1 + 1 = 33
    # Total = 1600 + 544 + 33 = 2177
    model = build_model_c(vocab_size=100, embedding_dim=16, hidden_dim=32)
    params = model.count_parameters()
    assert params["embedding_params"] == 1600
    assert params["dense_params"] == 577
    assert params["total_params"] == 2177


def test_model_padding_idx_zero_grad():
    model = NeuralTextClassifier(vocab_size=10, embedding_dim=4, pad_id=0)
    # Check padding_idx setting
    assert model.embedding.padding_idx == 0

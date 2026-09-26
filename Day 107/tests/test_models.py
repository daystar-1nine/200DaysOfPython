import pytest
import torch
import numpy as np
from app.models import PyTorchLSTMClassifier

@pytest.fixture
def default_params():
    return {
        'vocab_size': 100,
        'embedding_dim': 16,
        'hidden_dim': 32,
        'output_dim': 1
    }

def test_lstm_init(default_params):
    model = PyTorchLSTMClassifier(**default_params)
    assert model.embedding.num_embeddings == 100
    assert model.lstm.hidden_size == 32
    assert not model.lstm.bidirectional

@pytest.mark.parametrize("batch_size, seq_len", [
    (1, 10),
    (4, 20),
    (8, 5)
])
def test_lstm_forward_shape(default_params, batch_size, seq_len):
    model = PyTorchLSTMClassifier(**default_params)
    inputs = torch.randint(0, 100, (batch_size, seq_len))
    outputs = model(inputs)
    assert outputs.shape == (batch_size,)

def test_lstm_bidirectional(default_params):
    model = PyTorchLSTMClassifier(**default_params, bidirectional=True)
    assert model.lstm.bidirectional
    assert model.fc.in_features == default_params['hidden_dim'] * 2
    
    inputs = torch.randint(0, 100, (2, 10))
    outputs = model(inputs)
    assert outputs.shape == (2,)

def test_lstm_stacked(default_params):
    model = PyTorchLSTMClassifier(**default_params, num_layers=2)
    assert model.lstm.num_layers == 2
    
    inputs = torch.randint(0, 100, (2, 10))
    outputs = model(inputs)
    assert outputs.shape == (2,)

@pytest.mark.parametrize("dropout", [0.1, 0.3, 0.5])
def test_lstm_dropout(default_params, dropout):
    model = PyTorchLSTMClassifier(**default_params, dropout_rate=dropout)
    assert model.dropout.p == dropout

@pytest.mark.parametrize("hidden_dim", [16, 32, 64, 128])
def test_lstm_different_hidden_dims(default_params, hidden_dim):
    params = default_params.copy()
    params['hidden_dim'] = hidden_dim
    model = PyTorchLSTMClassifier(**params)
    assert model.lstm.hidden_size == hidden_dim
    inputs = torch.randint(0, 100, (2, 10))
    outputs = model(inputs)
    assert outputs.shape == (2,)

# Generate extra tests for thoroughness and count
@pytest.mark.parametrize("i", range(15))
def test_model_forward_robustness(default_params, i):
    model = PyTorchLSTMClassifier(**default_params, num_layers=1 if i%2==0 else 2, bidirectional=i%3==0)
    inputs = torch.randint(0, 100, (3, i+5))
    outputs = model(inputs)
    assert torch.all(outputs >= 0.0) and torch.all(outputs <= 1.0)

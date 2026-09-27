import pytest
import torch
from app.models.lstm import LSTMClassifier
from app.models.bilstm import BiLSTMClassifier

def test_lstm_build_and_forward():
    model = LSTMClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32, dense_dim=16)
    x = torch.randint(0, 100, (4, 12))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_bilstm_build_and_forward():
    model = BiLSTMClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32, dense_dim=16)
    x = torch.randint(0, 100, (4, 12))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_lstm_parameters_greater_than_gru():
    from app.models.gru import GRUClassifier
    lstm = LSTMClassifier(vocab_size=100, embedding_dim=32, hidden_dim=64)
    gru = GRUClassifier(vocab_size=100, embedding_dim=32, hidden_dim=64)
    # LSTM has 4 gates vs GRU's 3 gates
    assert lstm.count_parameters() > gru.count_parameters()

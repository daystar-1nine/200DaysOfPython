import pytest
import torch
from app.models.rnn import SimpleRNNClassifier

def test_rnn_build_and_forward():
    model = SimpleRNNClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32, dense_dim=16)
    x = torch.randint(0, 100, (4, 12))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_rnn_parameter_count():
    model = SimpleRNNClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32)
    params = model.count_parameters()
    assert params > 0
    assert isinstance(params, int)

@pytest.mark.parametrize("hidden_dim", [16, 32, 64])
def test_rnn_hidden_dimensions(hidden_dim):
    model = SimpleRNNClassifier(vocab_size=50, embedding_dim=16, hidden_dim=hidden_dim)
    x = torch.randint(0, 50, (2, 10))
    probs = model(x)
    assert probs.shape == (2,)

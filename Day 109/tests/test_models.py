import pytest
import torch
from app.models.gru import GRUClassifier
from app.models.gru_attention import GRUAttentionClassifier
from app.models.baseline import SimpleRNNClassifier, LSTMClassifier, TfidfLogisticRegressionBaseline

def test_gru_attention_classifier_build_and_forward():
    vocab_size = 80
    model = GRUAttentionClassifier(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32, attention_dim=32)
    x = torch.randint(1, vocab_size, (4, 10))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_gru_attention_returns_weights():
    vocab_size = 80
    model = GRUAttentionClassifier(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32, attention_dim=32)
    x = torch.randint(1, vocab_size, (3, 8))
    probs, weights = model(x, return_attention=True)
    assert probs.shape == (3,)
    assert weights.shape == (3, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-4)

def test_gru_attention_more_params_than_gru():
    vocab_size = 100
    gru = GRUClassifier(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)
    gru_att = GRUAttentionClassifier(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, attention_dim=64)
    assert gru_att.count_parameters() > gru.count_parameters()

def test_baseline_models_forward():
    vocab_size = 50
    rnn = SimpleRNNClassifier(vocab_size, embedding_dim=16, hidden_dim=32)
    lstm = LSTMClassifier(vocab_size, embedding_dim=16, hidden_dim=32)
    x = torch.randint(0, vocab_size, (2, 6))
    assert rnn(x).shape == (2,)
    assert lstm(x).shape == (2,)

def test_tfidf_baseline():
    clf = TfidfLogisticRegressionBaseline()
    texts = ["win cash prize now", "hello friend how are you", "free bonus lottery"]
    labels = [1, 0, 1]
    clf.fit(texts, labels)
    probs = clf.predict_proba(["win cash", "hello"])
    assert len(probs) == 2
    assert probs[0] > probs[1]
    assert clf.count_parameters() > 0

@pytest.mark.parametrize("batch_size, seq_len", [
    (1, 5),
    (2, 15),
    (5, 25)
])
def test_gru_attention_variable_batch_and_seq(batch_size, seq_len):
    model = GRUAttentionClassifier(vocab_size=60, embedding_dim=8, hidden_dim=16, attention_dim=16)
    x = torch.randint(0, 60, (batch_size, seq_len))
    probs, wts = model(x, return_attention=True)
    assert probs.shape == (batch_size,)
    assert wts.shape == (batch_size, seq_len)

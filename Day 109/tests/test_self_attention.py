import pytest
import numpy as np
from scratch.self_attention_numpy import SelfAttention

def test_self_attention_projections_and_shapes():
    seq_len, in_dim, att_dim = 6, 12, 24
    X = np.random.randn(seq_len, in_dim)
    
    layer = SelfAttention(input_dim=in_dim, attention_dim=att_dim)
    assert layer.Wq.shape == (in_dim, att_dim)
    assert layer.Wk.shape == (in_dim, att_dim)
    assert layer.Wv.shape == (in_dim, att_dim)
    
    context, weights = layer.forward(X)
    # Output dimension and sequence length preservation
    assert context.shape == (seq_len, att_dim)
    assert weights.shape == (seq_len, seq_len)
    assert np.allclose(weights.sum(axis=-1), 1.0)

def test_self_attention_batch_processing():
    batch, seq_len, in_dim, att_dim = 3, 5, 8, 16
    X = np.random.randn(batch, seq_len, in_dim)
    
    layer = SelfAttention(input_dim=in_dim, attention_dim=att_dim)
    context, weights = layer.forward(X)
    assert context.shape == (batch, seq_len, att_dim)
    assert weights.shape == (batch, seq_len, seq_len)
    for b in range(batch):
        assert np.allclose(weights[b].sum(axis=-1), 1.0)

@pytest.mark.parametrize("seq_len", [2, 5, 10, 20])
def test_self_attention_various_sequence_lengths(seq_len):
    in_dim, att_dim = 8, 16
    X = np.random.randn(seq_len, in_dim)
    layer = SelfAttention(input_dim=in_dim, attention_dim=att_dim)
    context, weights = layer.forward(X)
    assert context.shape == (seq_len, att_dim)
    assert weights.shape == (seq_len, seq_len)
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0)

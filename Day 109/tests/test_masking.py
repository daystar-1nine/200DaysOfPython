import pytest
import numpy as np
import torch
from scratch.attention_numpy import scaled_dot_product_attention
from scratch.self_attention_numpy import SelfAttention
from app.models.attention import AttentionLayer

def test_scaled_attention_masking_numpy():
    Q = np.random.randn(2, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(3, 8)
    
    # Mask out the 3rd key position (index 2)
    mask = np.array([
        [True, True, False],
        [True, True, False]
    ])
    
    out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert weights.shape == (2, 3)
    # The 3rd key weight should be strictly 0.0 or negligible
    assert np.allclose(weights[:, 2], 0.0, atol=1e-5)
    # Remaining valid keys should sum to 1.0
    assert np.allclose(weights[:, :2].sum(axis=-1), 1.0)

def test_self_attention_masking_numpy():
    seq_len = 4
    X = np.random.randn(seq_len, 8)
    # Last 2 positions are padding
    mask = np.array([1, 1, 0, 0])
    
    layer = SelfAttention(input_dim=8, attention_dim=16)
    context, weights = layer.forward(X, mask=mask)
    
    assert weights.shape == (seq_len, seq_len)
    # Columns 2 and 3 (padding positions) should receive 0 attention from all queries
    assert np.allclose(weights[:, 2:], 0.0, atol=1e-5)
    assert np.allclose(weights[:, :2].sum(axis=-1), 1.0)

def test_pytorch_attention_layer_masking():
    batch, seq_len, hidden_dim = 2, 5, 16
    inputs = torch.randn(batch, seq_len, hidden_dim)
    
    # Batch 0 has 3 valid tokens, Batch 1 has 2 valid tokens
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0]
    ], dtype=torch.long)
    
    attn_layer = AttentionLayer(hidden_dim=hidden_dim, attention_dim=32)
    context, weights = attn_layer(inputs, mask=mask)
    
    assert context.shape == (batch, hidden_dim)
    assert weights.shape == (batch, seq_len)
    
    # Check weights sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-4)
    
    # Check padding positions receive ~0
    weights_np = weights.detach().cpu().numpy()
    assert np.allclose(weights_np[0, 3:], 0.0, atol=1e-5)
    assert np.allclose(weights_np[1, 2:], 0.0, atol=1e-5)

def test_pytorch_attention_layer_without_mask():
    batch, seq_len, hidden_dim = 3, 4, 12
    inputs = torch.randn(batch, seq_len, hidden_dim)
    attn_layer = AttentionLayer(hidden_dim=hidden_dim)
    context, weights = attn_layer(inputs, mask=None)
    assert context.shape == (batch, hidden_dim)
    assert weights.shape == (batch, seq_len)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch), atol=1e-4)

"""
Unit tests for Transformer Encoder stack.
"""
import pytest
import torch
from app.models.transformer_encoder import TransformerEncoder


@pytest.mark.parametrize("num_layers", [1, 2, 4])
def test_transformer_encoder_layers_count(num_layers):
    d_model = 64
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=4,
        d_ff=128,
        dropout=0.0
    )
    x = torch.randn(2, 8, d_model)
    out, all_weights = encoder(x)
    assert out.shape == (2, 8, d_model)
    assert len(all_weights) == num_layers
    for w in all_weights:
        assert w.shape == (2, 4, 8, 8)


def test_transformer_encoder_sequence_length_preservation():
    encoder = TransformerEncoder(num_layers=3, d_model=32, num_heads=2, d_ff=64)
    for seq_len in [5, 12, 25]:
        x = torch.randn(1, seq_len, 32)
        out, _ = encoder(x)
        assert out.shape == (1, seq_len, 32)


def test_transformer_encoder_with_mask():
    encoder = TransformerEncoder(num_layers=2, d_model=32, num_heads=2, d_ff=64)
    x = torch.randn(2, 6, 32)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 4:] = True  # last 2 tokens are padded
    out, all_weights = encoder(x, mask=mask)
    assert out.shape == (2, 6, 32)
    for w in all_weights:
        # Padded key positions should have zero attention
        assert torch.allclose(w[:, :, :, 4:], torch.zeros_like(w[:, :, :, 4:]), atol=1e-5)

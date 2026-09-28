"""
Unit tests for MiniTransformerClassifier and pooling heads.
"""
import pytest
import torch
from app.models.classifier import MiniTransformerClassifier


def test_classifier_output_shape_and_probability_range():
    model = MiniTransformerClassifier(
        vocab_size=100,
        max_length=20,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1
    )
    model.eval()
    token_ids = torch.randint(0, 100, (4, 20))
    with torch.no_grad():
        probs, weights = model(token_ids)

    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert len(weights) == 1
    assert weights[0].shape == (4, 2, 20, 20)


@pytest.mark.parametrize("pooling", ["mask_mean", "mean", "first"])
def test_classifier_pooling_options(pooling):
    model = MiniTransformerClassifier(
        vocab_size=50,
        max_length=15,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        pooling=pooling
    )
    model.eval()
    token_ids = torch.randint(0, 50, (2, 15))
    with torch.no_grad():
        probs, _ = model(token_ids)
    assert probs.shape == (2, 1)


def test_classifier_parameter_counting():
    m1 = MiniTransformerClassifier(vocab_size=50, d_model=32, num_heads=2, d_ff=64, num_layers=1)
    m2 = MiniTransformerClassifier(vocab_size=50, d_model=32, num_heads=2, d_ff=64, num_layers=2)
    assert m2.count_parameters() > m1.count_parameters()


@pytest.mark.parametrize("ablation_kwargs", [
    {"use_positional_encoding": False},
    {"use_residual": False},
    {"use_layer_norm": False},
    {"use_padding_mask": False},
])
def test_classifier_ablation_forward_passes(ablation_kwargs):
    model = MiniTransformerClassifier(
        vocab_size=50,
        max_length=10,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        **ablation_kwargs
    )
    token_ids = torch.randint(0, 50, (2, 10))
    probs, _ = model(token_ids)
    assert probs.shape == (2, 1)


def test_classifier_training_step_reduces_loss():
    torch.manual_seed(42)
    model = MiniTransformerClassifier(
        vocab_size=20,
        max_length=6,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    x = torch.randint(0, 20, (8, 6))
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])

    # Initial loss
    out, _ = model(x)
    initial_loss = criterion(out, y).item()

    # Perform 5 gradient steps
    for _ in range(5):
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x)[0], y).item()
    assert final_loss < initial_loss


def test_classifier_eval_dropout_disabled():
    model = MiniTransformerClassifier(
        vocab_size=20,
        max_length=6,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        dropout=0.5
    )
    model.eval()
    x = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        p1, _ = model(x)
        p2, _ = model(x)
    assert torch.allclose(p1, p2)

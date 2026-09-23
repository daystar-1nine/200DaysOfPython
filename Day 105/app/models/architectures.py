"""
Model architecture factory functions for Day 105: Neural NLP & Text Classification.
Provides Model A (Baseline), Model B (Dense), and Model C (Regularized).
"""

from .embedding_classifier import NeuralTextClassifier


def build_model_a(
    vocab_size: int,
    embedding_dim: int = 64,
    pad_id: int = 0
) -> NeuralTextClassifier:
    """Model A: Embedding -> Masked Global Pool -> Linear(1) -> Output."""
    return NeuralTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=None,
        dropout_rate=0.0,
        pad_id=pad_id
    )


def build_model_b(
    vocab_size: int,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    pad_id: int = 0
) -> NeuralTextClassifier:
    """Model B: Embedding -> Masked Global Pool -> Dense(64) -> ReLU -> Output."""
    return NeuralTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=0.0,
        pad_id=pad_id
    )


def build_model_c(
    vocab_size: int,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    dropout_rate: float = 0.5,
    pad_id: int = 0
) -> NeuralTextClassifier:
    """Model C: Embedding -> Masked Global Pool -> Dense(64) -> ReLU -> Dropout(0.5) -> Output."""
    return NeuralTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        pad_id=pad_id
    )

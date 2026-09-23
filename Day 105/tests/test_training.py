"""
Unit tests for trainer, callbacks, and checkpointing.
Day 105: Neural NLP & Text Classification.
"""

import numpy as np
import pytest
import torch
from app.models.architectures import build_model_c
from app.training.callbacks import EarlyStopping
from app.training.checkpointing import ModelCheckpoint
from app.training.trainer import Trainer


def test_early_stopping_patience():
    es = EarlyStopping(patience=3, mode="min")
    assert not es(1.0)
    assert not es(1.1)  # strike 1
    assert not es(1.2)  # strike 2
    assert es(1.3)      # strike 3 -> triggers True


def test_early_stopping_reset_on_improvement():
    es = EarlyStopping(patience=2, mode="min")
    assert not es(1.0)
    assert not es(1.2)  # strike 1
    assert not es(0.8)  # improvement! Counter resets to 0
    assert not es(0.9)  # strike 1
    assert es(1.0)      # strike 2 -> triggers True


def test_model_checkpoint_save_and_load(tmp_path):
    save_file = tmp_path / "test_model.pt"
    cp = ModelCheckpoint(save_file, mode="min")
    m1 = build_model_c(vocab_size=10, embedding_dim=4, hidden_dim=4)

    # First step should save
    assert cp.step(0.5, m1)
    assert save_file.exists()

    # Step with worse loss should not save
    assert not cp.step(0.6, m1)

    # Step with better loss should save
    assert cp.step(0.3, m1)

    # Load best into a fresh model
    m2 = build_model_c(vocab_size=10, embedding_dim=4, hidden_dim=4)
    cp.load_best(m2)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_trainer_fit_convergence():
    torch.manual_seed(42)
    np.random.seed(42)
    X = np.ones((64, 10), dtype=np.int64)
    y = np.zeros((64,), dtype=np.float32)
    X[:32, :] = 2
    y[:32] = 1.0
    M = np.ones((64, 10), dtype=np.float32)

    loader = Trainer.create_dataloader(X, M, y, batch_size=16)
    model = build_model_c(vocab_size=25, embedding_dim=8, hidden_dim=8)
    trainer = Trainer(model, learning_rate=0.05)

    hist = trainer.fit(loader, loader, epochs=5, verbose=False)
    assert len(hist["train_loss"]) == 5
    assert hist["train_loss"][-1] < hist["train_loss"][0]

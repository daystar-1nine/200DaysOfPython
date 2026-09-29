"""
Unit tests for Day 112 training components (Data batching, Optimizer decay, Checkpointing, Trainer).
"""
import math
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from app.data.batching import get_batch
from app.data.dataset import prepare_data_tensors
from app.tokenizer.char_tokenizer import CharacterTokenizer
from app.training.optimizer import configure_adamw_optimizer
from app.training.checkpoint import save_checkpoint, load_checkpoint
from app.training.trainer import MiniGPTTrainer
from app.model.gpt import MiniGPT


@pytest.fixture
def tiny_dataset():
    # Tensor of 200 integers
    return torch.arange(200, dtype=torch.long)


@pytest.fixture
def tiny_model():
    return MiniGPT(
        vocab_size=32,
        context_length=16,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dropout=0.0
    )


class TestDataBatching:
    def test_get_batch_shapes(self, tiny_dataset):
        batch_size = 4
        context_length = 8
        x, y = get_batch(tiny_dataset, batch_size=batch_size, context_length=context_length)
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

    def test_get_batch_shifted_teacher_forcing(self, tiny_dataset):
        # In causal LM, y[b, t] must equal x[b, t + 1] when elements are consecutive
        # Since tiny_dataset is torch.arange(200), consecutive indices have consecutive values!
        x, y = get_batch(tiny_dataset, batch_size=5, context_length=10)
        for b in range(5):
            for t in range(9):
                assert y[b, t].item() == x[b, t + 1].item()

    def test_prepare_data_tensors_split(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        tok = CharacterTokenizer.from_text(text)
        train_data, val_data = prepare_data_tensors(text, tok, val_ratio=0.2)
        total_len = len(train_data) + len(val_data)
        assert total_len == len(text)
        assert len(val_data) == int(len(text) * 0.2)

    def test_get_batch_device_placement(self, tiny_dataset):
        device = torch.device("cpu")
        x, y = get_batch(tiny_dataset, batch_size=2, context_length=4, device=device)
        assert x.device == device
        assert y.device == device


class TestOptimizerConfiguration:
    def test_parameter_group_splitting(self, tiny_model):
        optimizer = configure_adamw_optimizer(tiny_model, weight_decay=0.1, learning_rate=1e-3)
        assert len(optimizer.param_groups) == 2
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

        # Verify no parameter is included in both groups
        decay_params = set(decay_group["params"])
        no_decay_params = set(no_decay_group["params"])
        assert len(decay_params.intersection(no_decay_params)) == 0

        # Verify all trainable parameters are covered
        all_trainable = {p for p in tiny_model.parameters() if p.requires_grad}
        assert decay_params.union(no_decay_params) == all_trainable

    def test_2d_weights_decayed_and_1d_biases_not_decayed(self, tiny_model):
        optimizer = configure_adamw_optimizer(tiny_model, weight_decay=0.05, learning_rate=1e-3)
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        for p in decay_group["params"]:
            assert p.dim() >= 2
        for p in no_decay_group["params"]:
            assert p.dim() < 2


class TestCheckpointing:
    def test_save_and_load_checkpoint_roundtrip(self, tmp_path, tiny_model):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "ckpt.pt"

        save_checkpoint(tiny_model, filepath=ckpt_path, optimizer=optimizer, step=42, loss=1.75)
        assert ckpt_path.exists()

        # Create fresh model with same architecture
        new_model = MiniGPT(
            vocab_size=tiny_model.vocab_size,
            context_length=tiny_model.context_length,
            embed_dim=tiny_model.embed_dim,
            num_heads=tiny_model.num_heads,
            num_layers=tiny_model.num_layers,
            dropout=0.0
        )
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

        ckpt_dict = load_checkpoint(ckpt_path, new_model, new_optimizer)
        assert ckpt_dict["step"] == 42
        assert math.isclose(ckpt_dict["loss"], 1.75, rel_tol=1e-4)

        # Check weights are bitwise identical
        for (n1, p1), (n2, p2) in zip(tiny_model.named_parameters(), new_model.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter {n1} did not match after loading"


class TestMiniGPTTrainer:
    def test_estimate_loss_returns_finite_losses(self, tiny_model, tiny_dataset):
        # Model vocab size is 32, so dataset values must be within [0, 31]
        bounded_dataset = torch.remainder(tiny_dataset, 32)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        trainer = MiniGPTTrainer(
            model=tiny_model,
            optimizer=optimizer,
            context_length=8,
            batch_size=4
        )
        loss = trainer.estimate_loss(bounded_dataset, eval_iters=3)
        assert not math.isnan(loss)
        assert loss > 0.0

    def test_single_train_step_updates_weights(self, tiny_model, tiny_dataset):
        bounded_dataset = torch.remainder(tiny_dataset, 32)
        train_data = bounded_dataset[:150]
        val_data = bounded_dataset[150:]

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-2)
        trainer = MiniGPTTrainer(
            model=tiny_model,
            optimizer=optimizer,
            context_length=8,
            batch_size=4
        )
        initial_weight = tiny_model.blocks[0].attn.c_attn.weight.clone()
        history = trainer.train(train_data, val_data, max_iters=3, eval_interval=10, eval_iters=1, verbose=False)
        updated_weight = tiny_model.blocks[0].attn.c_attn.weight
        assert not torch.equal(initial_weight, updated_weight)
        assert len(history["step"]) >= 1

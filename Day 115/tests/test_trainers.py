"""
Unit tests for RewardTrainer, DPOTrainer, CheckpointManager, and LR Schedulers.
Verifies training loops, gradient updates, schedulers, and checkpoint state persistence.
"""
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import torch.nn as nn

from app.config import (
    ModelConfig, RewardModelConfig, RewardTrainingConfig, DPOTrainingConfig
)
from app.models.minigpt import MiniGPTBackbone
from app.models.reward_model import RewardModel
from app.models.dpo_model import MiniGPTChat, DPOModel
from app.training.reward_trainer import RewardTrainer, get_cosine_schedule_with_warmup
from app.training.dpo_trainer import DPOTrainer
from app.training.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from app.data.collator import PreferenceCollator


@pytest.fixture
def sample_pairs():
    return [
        {"prompt": "Say hi", "chosen": "Hello! How can I help?", "rejected": "Bye.", "category": "helpfulness"},
        {"prompt": "What is 1+1?", "chosen": "1+1=2.", "rejected": "1+1=3.", "category": "correctness"},
        {"prompt": "Be brief", "chosen": "Sure.", "rejected": "I will proceed to explain why brevity is key at length.", "category": "conciseness"},
        {"prompt": "Tell me a fact", "chosen": "Water boils at 100C.", "rejected": "Water is red.", "category": "correctness"}
    ]


@pytest.fixture
def temp_checkpoint_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_checkpoint_save_and_load_functions(small_model_config, temp_checkpoint_dir):
    """Verifies low-level save_checkpoint and load_checkpoint functions."""
    model = MiniGPTChat(small_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_file = temp_checkpoint_dir / "test_model.pt"

    saved = save_checkpoint(ckpt_file, model, optimizer, epoch=2, step=40, metrics={"loss": 0.35})
    assert saved.exists()

    new_model = MiniGPTChat(small_model_config)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    meta = load_checkpoint(ckpt_file, new_model, new_opt)

    assert meta["epoch"] == 2
    assert meta["step"] == 40
    assert meta["metrics"]["loss"] == 0.35

    # Weights match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_manager_save_and_prune(small_model_config, temp_checkpoint_dir):
    """Verifies that CheckpointManager saves and rotates checkpoints to obey max_keep."""
    model = MiniGPTChat(small_model_config)
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, max_keep=2)

    for i in range(1, 5):
        manager.save_checkpoint(model, epoch=i, step=i * 10, metrics={"val_loss": 1.0 - i * 0.1})

    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) == 2


def test_cosine_schedule_with_warmup_calculation(small_model_config):
    """Verifies warmup ramp and cosine decay scheduler multiplier behavior."""
    model = MiniGPTChat(small_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=100, min_lr_ratio=0.1
    )

    # Step through warmup
    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # LR should increase during warmup
    assert lrs[-1] > lrs[0]

    # After warmup, LR decays
    for _ in range(80):
        scheduler.step()
    lr_end = optimizer.param_groups[0]["lr"]
    assert lr_end < lrs[-1]


def test_reward_trainer_initialization(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies RewardTrainer initializes dataloaders, optimizer, and scheduler properly."""
    rm = RewardModel(model_config=small_model_config)
    train_cfg = RewardTrainingConfig(num_epochs=1, batch_size=2, learning_rate=1e-3, warmup_steps=2)

    trainer = RewardTrainer(
        model=rm,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=train_cfg,
        output_dir=temp_checkpoint_dir
    )

    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert len(trainer.train_loader) == 2
    assert trainer.config.learning_rate == 1e-3


def test_reward_trainer_evaluate(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies RewardTrainer.evaluate computes val loss, accuracy, and margin."""
    rm = RewardModel(model_config=small_model_config)
    train_cfg = RewardTrainingConfig(num_epochs=1, batch_size=2)

    trainer = RewardTrainer(
        model=rm,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=train_cfg,
        output_dir=temp_checkpoint_dir
    )

    val_res = trainer.evaluate()
    assert "val_loss" in val_res
    assert "val_accuracy" in val_res
    assert "val_margin" in val_res
    assert 0.0 <= val_res["val_accuracy"] <= 100.0


def test_reward_trainer_full_train_loop(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies that RewardTrainer.train completes epochs and produces metrics history."""
    rm = RewardModel(model_config=small_model_config)
    train_cfg = RewardTrainingConfig(num_epochs=2, batch_size=2, eval_interval=1, learning_rate=1e-3)

    trainer = RewardTrainer(
        model=rm,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=train_cfg,
        output_dir=temp_checkpoint_dir,
        experiment_name="test_rm_exp"
    )

    result = trainer.train(verbose=False)
    assert "final_val_loss" in result
    assert "final_val_accuracy" in result
    assert len(trainer.history) > 0
    # Metrics file should be written
    metrics_file = temp_checkpoint_dir / "metrics" / "test_rm_exp.csv"
    assert metrics_file.exists()


def test_dpo_trainer_initialization(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies DPOTrainer initializes DPOModel and freezes reference weights."""
    policy = MiniGPTChat(small_model_config)
    ref = MiniGPTChat(small_model_config)
    dpo_model = DPOModel(policy_model=policy, reference_model=ref)

    dpo_cfg = DPOTrainingConfig(num_epochs=1, batch_size=2, learning_rate=1e-4, beta=0.1)

    trainer = DPOTrainer(
        dpo_model=dpo_model,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=dpo_cfg,
        output_dir=temp_checkpoint_dir
    )

    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    # Ensure reference model has requires_grad = False
    for p in trainer.dpo_model.reference.parameters():
        assert not p.requires_grad


def test_dpo_trainer_optimizer_only_tracks_policy(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies that only policy parameters are updated by optimizer."""
    policy = MiniGPTChat(small_model_config)
    ref = MiniGPTChat(small_model_config)
    dpo_model = DPOModel(policy_model=policy, reference_model=ref)
    dpo_cfg = DPOTrainingConfig(num_epochs=1, batch_size=2)

    trainer = DPOTrainer(
        dpo_model=dpo_model,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=dpo_cfg,
        output_dir=temp_checkpoint_dir
    )

    policy_ids = {id(p) for p in policy.parameters()}
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            assert id(p) in policy_ids


def test_dpo_trainer_evaluate(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies DPOTrainer.evaluate evaluates validation batch."""
    policy = MiniGPTChat(small_model_config)
    ref = MiniGPTChat(small_model_config)
    dpo_model = DPOModel(policy_model=policy, reference_model=ref)
    dpo_cfg = DPOTrainingConfig(num_epochs=1, batch_size=2)

    trainer = DPOTrainer(
        dpo_model=dpo_model,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=dpo_cfg,
        output_dir=temp_checkpoint_dir
    )

    val_res = trainer.evaluate()
    assert "val_loss" in val_res
    assert "val_accuracy" in val_res
    assert "val_margin" in val_res


def test_dpo_trainer_full_train_loop(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies that DPOTrainer.train completes epochs and saves output metrics."""
    policy = MiniGPTChat(small_model_config)
    ref = MiniGPTChat(small_model_config)
    dpo_model = DPOModel(policy_model=policy, reference_model=ref)
    dpo_cfg = DPOTrainingConfig(num_epochs=2, batch_size=2, eval_interval=1, learning_rate=1e-4)

    trainer = DPOTrainer(
        dpo_model=dpo_model,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=dpo_cfg,
        output_dir=temp_checkpoint_dir,
        experiment_name="test_dpo_exp"
    )

    result = trainer.train(verbose=False)
    assert "final_val_loss" in result
    assert "final_val_accuracy" in result
    assert len(trainer.history) > 0

    metrics_file = temp_checkpoint_dir / "metrics" / "test_dpo_exp.csv"
    assert metrics_file.exists()


def test_dpo_trainer_reference_weights_remain_unchanged(small_model_config, dummy_tokenizer, sample_pairs, temp_checkpoint_dir):
    """Verifies that reference model parameters remain unchanged during DPO training."""
    policy = MiniGPTChat(small_model_config)
    ref = MiniGPTChat(small_model_config)
    # Clone reference weights
    ref_weights = [p.clone() for p in ref.parameters()]

    dpo_model = DPOModel(policy_model=policy, reference_model=ref)
    dpo_cfg = DPOTrainingConfig(num_epochs=1, batch_size=2, eval_interval=1, learning_rate=1e-3)

    trainer = DPOTrainer(
        dpo_model=dpo_model,
        train_data=sample_pairs,
        val_data=sample_pairs,
        tokenizer=dummy_tokenizer,
        config=dpo_cfg,
        output_dir=temp_checkpoint_dir
    )

    trainer.train(verbose=False)

    for p_current, p_initial in zip(ref.parameters(), ref_weights):
        assert torch.equal(p_current, p_initial)

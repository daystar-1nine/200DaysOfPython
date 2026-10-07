"""
Unit tests for RewardModel architecture, pooling mechanisms,
Bradley-Terry pairwise preference loss, and ranking accuracy.
"""
import pytest
import torch
import torch.nn as nn

from app.config import ModelConfig, RewardModelConfig
from app.models.reward_model import RewardModel, compute_reward_loss
from app.evaluation.preference_score import evaluate_reward_model_preferences


def test_reward_config_defaults():
    cfg = RewardModelConfig()
    assert cfg.name == "MiniGPT-Reward"
    assert cfg.pooling_method == "last"
    assert cfg.head_hidden_dim is None


def test_reward_model_initialization(dummy_reward_model, small_model_config):
    assert dummy_reward_model.model_config == small_model_config
    assert isinstance(dummy_reward_model.reward_head, nn.Linear)
    assert dummy_reward_model.reward_head.in_features == small_model_config.embed_dim
    assert dummy_reward_model.reward_head.out_features == 1


def test_reward_model_forward_output_shape(dummy_reward_model):
    input_ids = torch.randint(0, 28, (3, 16))
    mask = torch.ones(3, 16)
    rewards = dummy_reward_model(input_ids, attention_mask=mask)
    assert rewards.shape == (3,)
    assert rewards.dtype == torch.float32


def test_reward_model_forward_pairwise(dummy_reward_model):
    c_ids = torch.randint(0, 28, (2, 12))
    c_mask = torch.ones(2, 12)
    r_ids = torch.randint(0, 28, (2, 10))
    r_mask = torch.ones(2, 10)

    rc, rr = dummy_reward_model.forward_pairwise(c_ids, c_mask, r_ids, r_mask)
    assert rc.shape == (2,)
    assert rr.shape == (2,)


def test_pool_representation_last_token(small_model_config):
    rm_cfg = RewardModelConfig(pooling_method="last")
    rm = RewardModel(model_config=small_model_config, reward_config=rm_cfg)

    # Sequence with padding at end
    hidden = torch.zeros(1, 5, small_model_config.embed_dim)
    # Put distinct vector at index 2 (last valid token)
    hidden[0, 2, :] = 3.14
    # Padding at indices 3 and 4
    mask = torch.tensor([[1, 1, 1, 0, 0]])

    pooled = rm.pool_representation(hidden, attention_mask=mask)
    assert pooled.shape == (1, small_model_config.embed_dim)
    torch.testing.assert_close(pooled[0], hidden[0, 2])


def test_pool_representation_mean(small_model_config):
    rm_cfg = RewardModelConfig(pooling_method="mean")
    rm = RewardModel(model_config=small_model_config, reward_config=rm_cfg)

    hidden = torch.ones(1, 4, small_model_config.embed_dim)
    mask = torch.tensor([[1, 1, 0, 0]])

    pooled = rm.pool_representation(hidden, attention_mask=mask)
    # Average of two vectors of all 1s is all 1s
    torch.testing.assert_close(pooled, torch.ones(1, small_model_config.embed_dim))


def test_compute_reward_loss_values():
    # When r_c == r_r, diff is 0, sigmoid(0) is 0.5, -ln(0.5) is ln(2) ~ 0.6931
    rc = torch.tensor([1.0, 2.0])
    rr = torch.tensor([1.0, 2.0])
    loss, metrics = compute_reward_loss(rc, rr)

    assert round(loss.item(), 4) == 0.6931
    assert metrics["accuracy"] == 0.0  # neither is strictly greater
    assert metrics["mean_margin"] == 0.0


def test_compute_reward_loss_with_margin():
    rc = torch.tensor([2.0])
    rr = torch.tensor([1.0])
    # diff = 2.0 - 1.0 - 1.0 = 0 -> -log(sigmoid(0)) = ln(2)
    loss, metrics = compute_reward_loss(rc, rr, margin=1.0)
    assert round(loss.item(), 4) == 0.6931
    assert metrics["accuracy"] == 1.0
    assert metrics["mean_margin"] == 1.0


def test_compute_reward_loss_accuracy_all_correct():
    rc = torch.tensor([5.0, 3.0, 4.0])
    rr = torch.tensor([1.0, 0.5, 2.0])
    loss, metrics = compute_reward_loss(rc, rr)
    assert metrics["accuracy"] == 1.0
    assert metrics["mean_margin"] > 0


def test_compute_reward_loss_accuracy_all_wrong():
    rc = torch.tensor([1.0, 0.5])
    rr = torch.tensor([3.0, 2.0])
    loss, metrics = compute_reward_loss(rc, rr)
    assert metrics["accuracy"] == 0.0
    assert metrics["mean_margin"] < 0


def test_reward_model_backward_pass(dummy_reward_model):
    dummy_reward_model.train()
    c_ids = torch.randint(0, 28, (2, 8))
    r_ids = torch.randint(0, 28, (2, 8))
    c_mask = torch.ones(2, 8)
    r_mask = torch.ones(2, 8)

    rc, rr = dummy_reward_model.forward_pairwise(c_ids, c_mask, r_ids, r_mask)
    loss, _ = compute_reward_loss(rc, rr)
    loss.backward()

    # Verify gradients exist on backbone and head
    for name, param in dummy_reward_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"


def test_mlp_reward_head_option(small_model_config):
    rm_cfg = RewardModelConfig(head_hidden_dim=64)
    rm = RewardModel(model_config=small_model_config, reward_config=rm_cfg)
    assert isinstance(rm.reward_head, nn.Sequential)

    x = torch.randint(0, 28, (2, 10))
    m = torch.ones(2, 10)
    out = rm(x, attention_mask=m)
    assert out.shape == (2,)


def test_evaluate_reward_model_preferences(dummy_reward_model, dummy_tokenizer, sample_preference_batch):
    res = evaluate_reward_model_preferences(
        dummy_reward_model, sample_preference_batch, dummy_tokenizer, batch_size=2
    )
    assert "accuracy" in res
    assert "mean_margin" in res
    assert "category_breakdown" in res
    assert res["total_evaluated"] == len(sample_preference_batch)

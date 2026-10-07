"""
Unit tests for MiniGPTChat, response-only log-probability engine,
Direct Preference Optimization (DPO) loss, and DPOModel wrapper.
"""
import pytest
import torch
import torch.nn as nn

from app.config import ModelConfig, GenerationConfig
from app.models.dpo_model import MiniGPTChat, DPOModel, compute_dpo_loss
from app.evaluation.preference_score import evaluate_policy_preferences


def test_minigpt_chat_forward_logits_shape(dummy_mini_chat, small_model_config):
    x = torch.randint(0, small_model_config.vocab_size, (2, 10))
    logits, loss = dummy_mini_chat(x)
    assert logits.shape == (2, 10, small_model_config.vocab_size)
    assert loss is None


def test_minigpt_chat_forward_loss(dummy_mini_chat, small_model_config):
    x = torch.randint(0, small_model_config.vocab_size, (2, 10))
    labels = x.clone()
    logits, loss = dummy_mini_chat(x, labels=labels)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_minigpt_chat_weight_tying(dummy_mini_chat):
    assert dummy_mini_chat.lm_head.weight.data_ptr() == dummy_mini_chat.backbone.tok_emb.weight.data_ptr()


def test_context_window_overflow_raises(dummy_mini_chat, small_model_config):
    oversized = torch.randint(0, small_model_config.vocab_size, (1, small_model_config.context_length + 5))
    with pytest.raises(ValueError, match="exceeds maximum context length"):
        dummy_mini_chat(oversized)


def test_get_response_log_probs_output_shape(dummy_mini_chat, small_model_config):
    x = torch.randint(0, small_model_config.vocab_size, (3, 10))
    labels = x.clone()
    labels[:, :4] = -100  # Prompt tokens masked

    logps = dummy_mini_chat.get_response_log_probs(x, labels=labels)
    assert logps.shape == (3,)
    assert logps.dtype == torch.float32
    # Log probabilities are non-positive
    assert (logps <= 0.0).all()


def test_get_response_log_probs_requires_labels(dummy_mini_chat, small_model_config):
    x = torch.randint(0, small_model_config.vocab_size, (2, 8))
    with pytest.raises(ValueError, match="Labels are required"):
        dummy_mini_chat.get_response_log_probs(x, labels=None)


def test_get_response_log_probs_ignores_masked_prompt_tokens(dummy_mini_chat, small_model_config):
    x = torch.randint(0, small_model_config.vocab_size, (1, 8))
    # labels1 has prompt tokens masked
    labels1 = x.clone()
    labels1[0, :5] = -100

    # labels2 masks prompt AND changes values in prompt portion of x2
    x2 = x.clone()
    x2[0, :5] = torch.randint(0, small_model_config.vocab_size, (5,))

    # Log probs should only be evaluated on the unmasked response portion
    logp1 = dummy_mini_chat.get_response_log_probs(x, labels=labels1)
    # The active response tokens are pos 5, 6, 7 (shifted targets 5, 6, 7)
    assert not torch.isnan(logp1)


def test_autoregressive_generate(dummy_mini_chat):
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
    out = dummy_mini_chat.generate(prompt, generation_config=cfg)

    assert out.size(0) == 1
    assert out.size(1) == 3 + 5
    assert out[0, :3].tolist() == [1, 2, 3]


def test_greedy_generation_is_deterministic(dummy_mini_chat):
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=6, do_sample=False)
    out1 = dummy_mini_chat.generate(prompt, generation_config=cfg)
    out2 = dummy_mini_chat.generate(prompt, generation_config=cfg)
    assert torch.equal(out1, out2)


def test_compute_dpo_loss_sigmoid():
    # Symmetric identical log probabilities: margin = 0 -> loss = -ln(0.5) ~ 0.6931
    pi_c = torch.tensor([-2.0])
    pi_r = torch.tensor([-3.0])
    ref_c = torch.tensor([-2.0])
    ref_r = torch.tensor([-3.0])

    loss, metrics = compute_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1, loss_type="sigmoid")
    assert round(loss.item(), 4) == 0.6931
    assert metrics["accuracy"] == 0.0
    assert metrics["reward_margin"] == 0.0


def test_compute_dpo_loss_accuracy_when_chosen_preferred():
    pi_c = torch.tensor([-1.0])
    pi_r = torch.tensor([-4.0])
    ref_c = torch.tensor([-2.0])
    ref_r = torch.tensor([-2.0])

    # pi_c - ref_c = 1.0; pi_r - ref_r = -2.0 -> margin = beta * (1 - (-2)) = beta * 3 = 0.3 > 0
    loss, metrics = compute_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1, loss_type="sigmoid")
    assert metrics["accuracy"] == 1.0
    assert metrics["reward_margin"] > 0
    assert loss.item() < 0.6931


def test_compute_dpo_loss_hinge_variant():
    pi_c = torch.tensor([-1.0])
    pi_r = torch.tensor([-3.0])
    ref_c = torch.tensor([-2.0])
    ref_r = torch.tensor([-2.0])

    loss, metrics = compute_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1, loss_type="hinge")
    assert loss.item() >= 0.0
    assert "accuracy" in metrics


def test_compute_dpo_loss_ipo_variant():
    pi_c = torch.tensor([-2.0])
    pi_r = torch.tensor([-3.0])
    ref_c = torch.tensor([-2.0])
    ref_r = torch.tensor([-3.0])

    loss, metrics = compute_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1, loss_type="ipo")
    assert loss.item() > 0.0


def test_unsupported_dpo_loss_type_raises():
    t = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="Unsupported DPO loss_type"):
        compute_dpo_loss(t, t, t, t, loss_type="unknown")


def test_dpo_model_reference_strictly_frozen(dummy_dpo_model):
    for p in dummy_dpo_model.reference.parameters():
        assert p.requires_grad is False
    for p in dummy_dpo_model.policy.parameters():
        assert p.requires_grad is True


def test_dpo_model_forward_pairwise_gradients(dummy_dpo_model, small_model_config):
    batch = {
        "chosen_input_ids": torch.randint(0, small_model_config.vocab_size, (2, 8)),
        "chosen_attention_mask": torch.ones(2, 8),
        "chosen_labels": torch.full((2, 8), -100),
        "rejected_input_ids": torch.randint(0, small_model_config.vocab_size, (2, 8)),
        "rejected_attention_mask": torch.ones(2, 8),
        "rejected_labels": torch.full((2, 8), -100)
    }
    # Unmask last 3 tokens
    batch["chosen_labels"][:, -3:] = batch["chosen_input_ids"][:, -3:]
    batch["rejected_labels"][:, -3:] = batch["rejected_input_ids"][:, -3:]

    loss, metrics = dummy_dpo_model.forward_pairwise(batch, beta=0.1)
    loss.backward()

    # Policy must have gradients
    for name, p in dummy_dpo_model.policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Policy param {name} should have gradients"

    # Reference must NOT have gradients
    for p in dummy_dpo_model.reference.parameters():
        assert p.grad is None


def test_evaluate_policy_preferences(dummy_mini_chat, dummy_tokenizer, sample_preference_batch):
    res = evaluate_policy_preferences(dummy_mini_chat, sample_preference_batch, dummy_tokenizer)
    assert "accuracy" in res
    assert "mean_log_margin" in res
    assert res["total_evaluated"] == len(sample_preference_batch)

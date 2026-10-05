"""
Unit tests for Low-Rank Adaptation (LoRA) mechanics:
LoRALinear, zero-initialization identity, scaling factor, weight merging/unmerging,
gradient isolation, and model-level adapter injection.
"""
import pytest
import torch
import torch.nn as nn

from app.config import ChatModelConfig, LoRAConfig
from app.model.minigpt_chat import LoRALinear, MiniGPTChat
from app.training.checkpoint import save_sft_checkpoint, load_sft_checkpoint


def test_lora_config_defaults():
    cfg = LoRAConfig()
    assert cfg.r == 8
    assert cfg.lora_alpha == 16.0
    assert "c_attn" in cfg.target_modules
    assert "c_proj" in cfg.target_modules


def test_lora_linear_initialization_zero_identity():
    """At initialization, B=0, so B @ A = 0, meaning LoRALinear(x) == Linear(x)."""
    base = nn.Linear(32, 64, bias=True)
    lora = LoRALinear(base, r=4, lora_alpha=8.0)

    # lora_A should be non-zero, lora_B must be exactly zero
    assert not torch.all(lora.lora_A == 0)
    assert torch.all(lora.lora_B == 0)

    x = torch.randn(2, 5, 32)
    base_out = base(x)
    lora_out = lora(x)

    torch.testing.assert_close(base_out, lora_out, atol=1e-6, rtol=1e-5)


def test_lora_linear_parameter_freezing():
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, lora_alpha=8.0)

    assert lora.base_linear.weight.requires_grad is False
    if lora.base_linear.bias is not None:
        assert lora.base_linear.bias.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True


def test_lora_linear_scaling_factor():
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, lora_alpha=12.0)
    assert lora.scaling == 3.0  # 12.0 / 4


def test_lora_linear_backward_gradients_only_on_adapters():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=2, lora_alpha=4.0)
    x = torch.randn(2, 8)
    out = lora(x).sum()
    out.backward()

    assert lora.base_linear.weight.grad is None
    if lora.base_linear.bias is not None:
        assert lora.base_linear.bias.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_lora_weight_merging_and_unmerging():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=2, lora_alpha=4.0)
    # Set non-zero weights in lora_B for testing merge
    nn.init.normal_(lora.lora_B, std=0.1)

    x = torch.randn(2, 8)
    unmerged_out = lora(x)

    # Merge weights into base weight
    lora.merge_weights()
    assert lora.merged is True
    merged_out = lora(x)

    # Merged forward must produce mathematically identical result
    torch.testing.assert_close(unmerged_out, merged_out, atol=1e-5, rtol=1e-5)

    # Unmerge weights
    lora.unmerge_weights()
    assert lora.merged is False
    unmerged_again_out = lora(x)

    torch.testing.assert_close(unmerged_out, unmerged_again_out, atol=1e-5, rtol=1e-5)


def test_double_merge_prevention():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=2, lora_alpha=4.0)
    nn.init.normal_(lora.lora_B, std=0.1)
    base_weight_before = lora.base_linear.weight.clone()

    lora.merge_weights()
    weight_after_first = lora.base_linear.weight.clone()
    # Second call should be a no-op because lora.merged == True
    lora.merge_weights()
    weight_after_second = lora.base_linear.weight.clone()

    torch.testing.assert_close(weight_after_first, weight_after_second)
    assert not torch.equal(base_weight_before, weight_after_first)


def test_model_apply_lora():
    cfg = ChatModelConfig(vocab_size=32, context_length=32, embed_dim=32, num_heads=2, num_layers=2)
    model = MiniGPTChat(cfg)

    orig_total_trainable = model.count_parameters()["trainable"]
    stats = model.apply_lora(r=4, lora_alpha=8.0, target_modules=["c_attn"])
    lora_trainable = stats["trainable"]

    # Trainable parameters should drop significantly
    assert lora_trainable < orig_total_trainable
    assert lora_trainable > 0

    # Verify target layers were converted to LoRALinear
    for block in model.blocks:
        assert isinstance(block.attn.c_attn, LoRALinear)
        # c_proj was not in target_modules, should remain standard Linear
        assert type(block.attn.c_proj) is nn.Linear


def test_lora_checkpoint_save_and_load(tmp_path):
    cfg = ChatModelConfig(vocab_size=32, context_length=32, embed_dim=32, num_heads=2, num_layers=2)
    model1 = MiniGPTChat(cfg)
    model1.apply_lora(r=4, lora_alpha=8.0, target_modules=["c_attn"])

    # Modify adapter weights
    for p in model1.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)

    ckpt_path = tmp_path / "lora_adapter.pt"
    save_sft_checkpoint(ckpt_path, model1, lora_only=True)
    assert ckpt_path.exists()

    # Load adapter into fresh model with same architecture
    model2 = MiniGPTChat(cfg)
    model2.apply_lora(r=4, lora_alpha=8.0, target_modules=["c_attn"])
    load_sft_checkpoint(ckpt_path, model2)

    # Verify adapter weights match
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if "lora_" in n1:
            torch.testing.assert_close(p1, p2)

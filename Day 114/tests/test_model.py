"""
Unit tests for MiniGPTChat architecture, Pre-LN blocks, causal masking,
weight tying, and autoregressive generation mechanics.
"""
import pytest
import torch
import torch.nn.functional as F

from app.config import ChatModelConfig
from app.model.minigpt_chat import (
    MiniGPTChat,
    TransformerBlock,
    CausalSelfAttention,
    GELUMLP
)


@pytest.fixture
def small_config():
    return ChatModelConfig(
        vocab_size=32,
        context_length=32,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        tie_weights=True
    )


@pytest.fixture
def small_model(small_config):
    torch.manual_seed(42)
    return MiniGPTChat(small_config)


def test_model_config_defaults():
    cfg = ChatModelConfig(vocab_size=100)
    assert cfg.vocab_size == 100
    assert cfg.context_length == 128
    assert cfg.embed_dim == 128
    assert cfg.num_heads == 4
    assert cfg.num_layers == 4
    assert cfg.tie_weights is True


def test_model_initialization(small_model, small_config):
    assert len(small_model.blocks) == small_config.num_layers
    assert small_model.tok_emb.num_embeddings == small_config.vocab_size
    assert small_model.pos_emb.num_embeddings == small_config.context_length
    assert small_model.lm_head.out_features == small_config.vocab_size


def test_weight_tying_enabled(small_config):
    small_config.tie_weights = True
    model = MiniGPTChat(small_config)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_weight_tying_disabled(small_config):
    small_config.tie_weights = False
    model = MiniGPTChat(small_config)
    assert model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr()


def test_causal_mask_structure(small_config):
    attn = CausalSelfAttention(small_config)
    mask = attn.causal_mask.squeeze()  # [T, T]
    T = small_config.context_length

    # Lower triangular should be 1, upper triangular should be 0
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert mask[i, j].item() == 1
            else:
                assert mask[i, j].item() == 0


def test_causal_attention_future_invariance(small_config):
    """Changing future tokens must not affect current token's attention outputs."""
    torch.manual_seed(42)
    attn = CausalSelfAttention(small_config).eval()

    x1 = torch.randn(1, 6, small_config.embed_dim)
    x2 = x1.clone()
    # Modify future tokens (positions 3, 4, 5)
    x2[:, 3:, :] = torch.randn(1, 3, small_config.embed_dim)

    out1 = attn(x1)
    out2 = attn(x2)

    # Position 0, 1, 2 must produce identical outputs
    torch.testing.assert_close(out1[:, :3, :], out2[:, :3, :], atol=1e-6, rtol=1e-5)


def test_transformer_block_forward(small_config):
    block = TransformerBlock(small_config).eval()
    x = torch.randn(2, 8, small_config.embed_dim)
    out = block(x)
    assert out.shape == x.shape


def test_gelu_mlp_dimensions(small_config):
    mlp = GELUMLP(small_config)
    assert mlp.c_fc.in_features == small_config.embed_dim
    assert mlp.c_fc.out_features == 4 * small_config.embed_dim
    assert mlp.c_proj.in_features == 4 * small_config.embed_dim
    assert mlp.c_proj.out_features == small_config.embed_dim


def test_forward_logits_shape(small_model):
    input_ids = torch.randint(0, 32, (2, 10))
    logits, loss = small_model(input_ids)
    assert logits.shape == (2, 10, 32)
    assert loss is None


def test_forward_with_targets_returns_loss(small_model):
    input_ids = torch.randint(0, 32, (2, 10))
    targets = input_ids.clone()
    logits, loss = small_model(input_ids, targets=targets)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0.0


def test_loss_ignores_minus_100(small_model):
    input_ids = torch.randint(0, 32, (1, 8))
    # All targets except the last token are -100
    targets = torch.full_like(input_ids, -100)
    targets[0, -1] = 5

    logits, loss = small_model(input_ids, targets=targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_context_window_overflow_raises_error(small_model, small_config):
    oversized = torch.randint(0, 32, (1, small_config.context_length + 5))
    with pytest.raises(ValueError, match="exceeds context window"):
        small_model(oversized)


def test_parameter_count_methods(small_model):
    counts = small_model.count_parameters()
    assert "total" in counts
    assert "trainable" in counts
    assert "frozen" in counts
    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]


def test_backward_pass_gradients(small_model):
    small_model.train()
    input_ids = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    _, loss = small_model(input_ids, targets=targets)
    loss.backward()

    for name, param in small_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"


def test_generate_output_length(small_model):
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = small_model.generate(prompt, max_new_tokens=5, do_sample=False)
    assert out.size(0) == 1
    assert out.size(1) == 3 + 5
    # Prompt prefix preserved
    assert out[0, :3].tolist() == [1, 2, 3]


def test_generate_greedy_deterministic(small_model):
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out1 = small_model.generate(prompt, max_new_tokens=8, do_sample=False)
    out2 = small_model.generate(prompt, max_new_tokens=8, do_sample=False)
    assert torch.equal(out1, out2)


def test_generate_stops_at_stop_token(small_model):
    stop_id = 9
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    # Force model to always predict stop_id
    with torch.no_grad():
        small_model.lm_head.weight.fill_(0.0)
        # Give huge bias to stop_id
        bias = torch.zeros(small_model.config.vocab_size)
        bias[stop_id] = 100.0
        small_model.lm_head.bias = torch.nn.Parameter(bias)

    out = small_model.generate(prompt, max_new_tokens=10, stop_token_id=stop_id, do_sample=False)
    # Should stop on the very first generated step
    assert out.size(1) == 4
    assert out[0, -1].item() == stop_id

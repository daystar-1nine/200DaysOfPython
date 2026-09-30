"""
Unit tests for ScalableMiniGPT model components:
- CausalSelfAttention
- GELUMLP
- TransformerBlock
- ScalableMiniGPT architecture
- Parameter counting and weight tying
- Forward and backward gradient propagation
"""
import pytest
import torch
import torch.nn as nn
from app.config import ModelConfig
from app.model.minigpt import ScalableMiniGPT, TransformerBlock, CausalSelfAttention, GELUMLP


class TestCausalSelfAttention:
    def test_attention_output_shape(self):
        B, T, C = 2, 8, 32
        x = torch.randn(B, T, C)
        attn = CausalSelfAttention(embed_dim=C, num_heads=4, context_length=16)
        out, _ = attn(x)
        assert out.shape == (B, T, C)

    def test_attention_indivisible_heads_raises_error(self):
        with pytest.raises(ValueError, match="divisible"):
            CausalSelfAttention(embed_dim=30, num_heads=4, context_length=16)

    def test_attention_causal_mask_zeros_future_weights(self):
        B, T, C = 1, 6, 32
        x = torch.randn(B, T, C)
        attn = CausalSelfAttention(embed_dim=C, num_heads=2, context_length=16, dropout=0.0)
        attn.eval()
        _, weights = attn(x, return_weights=True)

        assert weights.shape == (1, 2, 6, 6)
        # Upper triangle must be strictly 0.0
        upper = torch.triu(weights[0, 0], diagonal=1)
        assert torch.all(upper == 0.0)

        # Lower triangle + diagonal must sum to 1.0 along the row
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_causality_invariance(self):
        """Modifying future tokens at t > k must not alter representations at t <= k."""
        B, T, C = 1, 5, 32
        attn = CausalSelfAttention(embed_dim=C, num_heads=2, context_length=8, dropout=0.0)
        attn.eval()

        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        x2[:, 3:, :] = torch.randn(B, 2, C) * 5.0  # Mutate last 2 tokens

        with torch.no_grad():
            out1, _ = attn(x1)
            out2, _ = attn(x2)

        # Steps 0, 1, 2 must be identical
        assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5)
        # Steps 3, 4 must differ
        assert not torch.allclose(out1[:, 3:, :], out2[:, 3:, :], atol=1e-3)


class TestGELUMLPAndBlock:
    def test_mlp_expansion_shape(self):
        B, T, C = 2, 4, 32
        x = torch.randn(B, T, C)
        mlp = GELUMLP(embed_dim=C, dropout=0.0, ff_multiplier=4)
        out = mlp(x)
        assert out.shape == (B, T, C)
        assert mlp.c_fc.out_features == 4 * C

    def test_transformer_block_preserves_shape(self):
        B, T, C = 2, 6, 32
        x = torch.randn(B, T, C)
        block = TransformerBlock(embed_dim=C, num_heads=4, context_length=16)
        out, _ = block(x)
        assert out.shape == (B, T, C)

    def test_block_gradients_flow(self):
        x = torch.randn(2, 4, 32, requires_grad=True)
        block = TransformerBlock(embed_dim=32, num_heads=2, context_length=8)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert block.mlp.c_fc.weight.grad is not None
        assert block.attn.c_attn.weight.grad is not None


class TestScalableMiniGPT:
    def test_model_forward_without_targets(self, tiny_model):
        x = torch.randint(0, 30, (2, 8))
        logits, loss, _ = tiny_model(x)
        assert logits.shape == (2, 8, tiny_model.vocab_size)
        assert loss is None

    def test_model_forward_with_targets(self, tiny_model):
        x = torch.randint(0, 30, (2, 8))
        y = torch.randint(0, 30, (2, 8))
        logits, loss, _ = tiny_model(x, targets=y)
        assert logits.shape == (2, 8, tiny_model.vocab_size)
        assert loss is not None
        assert loss.item() > 0.0

    def test_sequence_exceeding_context_length_raises(self, tiny_model):
        long_x = torch.randint(0, 30, (1, tiny_model.context_length + 5))
        with pytest.raises(ValueError, match="exceeds context window"):
            tiny_model(long_x)

    def test_weight_tying_active(self, tiny_model):
        assert tiny_model.lm_head.weight is tiny_model.token_embeddings.weight

    def test_weight_tying_disabled(self):
        cfg = ModelConfig(vocab_size=32, embed_dim=32, num_heads=2, num_layers=2, tie_weights=False)
        model = ScalableMiniGPT(cfg)
        assert model.lm_head.weight is not model.token_embeddings.weight

    def test_count_parameters_accuracy(self, tiny_model):
        counts = tiny_model.count_parameters()
        assert counts["total"] == sum(p.numel() for p in tiny_model.parameters())
        assert counts["trainable"] == counts["total"]
        assert counts["embeddings"] > 0
        assert counts["blocks"] > 0
        assert counts["final_layernorm"] > 0
        assert counts["lm_head"] == 0  # Tied

    def test_single_token_forward_pass(self, tiny_model):
        x = torch.tensor([[5]])
        logits, loss, _ = tiny_model(x)
        assert logits.shape == (1, 1, tiny_model.vocab_size)
        assert loss is None

    def test_backprop_accumulates_gradients(self, tiny_model):
        x = torch.randint(0, 30, (2, 4))
        y = torch.randint(0, 30, (2, 4))
        _, loss, _ = tiny_model(x, targets=y)
        loss.backward()

        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} had no gradient!"

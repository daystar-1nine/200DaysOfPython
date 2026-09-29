"""
Unit tests for MLP, TransformerBlock, MiniGPT, and MiniGPTScratch.
"""
import pytest
import torch
from app.model.block import MLP, TransformerBlock
from app.model.gpt import MiniGPT
from scratch.gpt_model import MiniGPTScratch


class TestTransformerBlockAndMLP:
    def test_mlp_expansion_and_shape(self):
        B, T, C = 2, 4, 32
        x = torch.randn(B, T, C)
        mlp = MLP(embed_dim=C, dropout=0.0)
        out = mlp(x)
        assert out.shape == (B, T, C)
        assert mlp.c_fc.out_features == 4 * C

    def test_transformer_block_preserves_shape(self):
        B, T, C = 2, 8, 32
        x = torch.randn(B, T, C)
        block = TransformerBlock(embed_dim=C, num_heads=4, context_length=16)
        out, weights = block(x, return_weights=True)
        assert out.shape == (B, T, C)
        assert weights.shape == (B, 4, 8, 8)

    def test_block_gradients_flow(self):
        x = torch.randn(2, 4, 32, requires_grad=True)
        block = TransformerBlock(embed_dim=32, num_heads=2, context_length=8)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert block.mlp.c_fc.weight.grad is not None
        assert block.attn.c_attn.weight.grad is not None

    def test_pre_layer_norm_architecture(self):
        B, T, C = 2, 4, 32
        block = TransformerBlock(embed_dim=C, num_heads=2, context_length=8)
        assert hasattr(block, "ln_1")
        assert hasattr(block, "ln_2")
        assert hasattr(block, "attn")
        assert hasattr(block, "mlp")


class TestMiniGPT:
    def test_weight_tying_shares_memory(self):
        model = MiniGPT(vocab_size=50, context_length=16, embed_dim=32, num_heads=2, num_layers=2, tie_weights=True)
        assert model.lm_head.weight is model.token_embeddings.weight

    def test_weight_tying_disabled(self):
        model = MiniGPT(vocab_size=50, context_length=16, embed_dim=32, num_heads=2, num_layers=2, tie_weights=False)
        assert model.lm_head.weight is not model.token_embeddings.weight

    def test_forward_pass_without_targets(self, tiny_gpt_model):
        x = torch.randint(0, 30, (2, 8))
        logits, loss, _ = tiny_gpt_model(x)
        assert logits.shape == (2, 8, 32)
        assert loss is None

    def test_forward_pass_with_targets(self, tiny_gpt_model):
        x = torch.randint(0, 30, (2, 8))
        y = torch.randint(0, 30, (2, 8))
        logits, loss, _ = tiny_gpt_model(x, targets=y)
        assert logits.shape == (2, 8, 32)
        assert loss is not None
        assert loss.item() > 0.0

    def test_single_token_forward_pass(self, tiny_gpt_model):
        x = torch.randint(0, 30, (1, 1))
        logits, loss, _ = tiny_gpt_model(x)
        assert logits.shape == (1, 1, 32)
        assert loss is None

    def test_positional_embeddings_shape(self, tiny_gpt_model):
        assert tiny_gpt_model.position_embeddings.weight.shape == (16, 32)
        assert tiny_gpt_model.token_embeddings.weight.shape == (32, 32)

    def test_exceeding_context_length_raises_error(self, tiny_gpt_model):
        # tiny_gpt_model has context_length=16
        long_input = torch.randint(0, 30, (1, 20))
        with pytest.raises(ValueError):
            tiny_gpt_model(long_input)

    def test_parameter_counting(self, tiny_gpt_model):
        counts = tiny_gpt_model.count_parameters()
        assert "total" in counts
        assert "trainable" in counts
        assert counts["total"] == counts["trainable"]
        assert counts["total"] > 10_000

    def test_scratch_model_parity(self):
        ms = MiniGPTScratch(vocab_size=30, context_length=16, embed_dim=32, num_heads=2, num_layers=2)
        x = torch.randint(0, 30, (2, 8))
        y = torch.randint(0, 30, (2, 8))
        logits, loss = ms(x, targets=y)
        assert logits.shape == (2, 8, 30)
        assert loss is not None

"""
Unit tests for text generation loops and decoding sampling strategies.
"""
import pytest
import torch
import numpy as np
from app.generation.generate import generate_tokens, generate_text_from_prompt
from app.generation.sampling import sample_next_token, generate_with_strategy
from scratch.generation import generate_greedy_scratch
from scratch.sampling import (
    sample_temperature_np,
    sample_top_k_np,
    sample_top_p_np
)


class TestTokenGeneration:
    def test_output_length_and_prefix_preservation(self, tiny_gpt_model):
        prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        max_new = 8
        out = generate_tokens(tiny_gpt_model, prompt, max_new_tokens=max_new, context_length=16)

        assert out.shape == (1, 4 + max_new)
        assert (out[:, :4] == prompt).all()

    def test_greedy_is_deterministic(self, tiny_gpt_model):
        prompt = torch.tensor([[5, 10, 15]], dtype=torch.long)
        out1 = generate_tokens(tiny_gpt_model, prompt, max_new_tokens=6, context_length=16)
        out2 = generate_tokens(tiny_gpt_model, prompt, max_new_tokens=6, context_length=16)
        assert (out1 == out2).all()

    def test_context_window_sliding_does_not_overflow(self, tiny_gpt_model):
        # tiny_gpt_model has context_length=16. Generating 25 tokens forces context window cropping!
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        out = generate_tokens(tiny_gpt_model, prompt, max_new_tokens=25, context_length=16)
        assert out.shape == (1, 27)

    def test_generate_text_from_prompt_helper(self, tiny_gpt_model, char_tokenizer):
        text = generate_text_from_prompt(
            model=tiny_gpt_model,
            tokenizer=char_tokenizer,
            prompt="First",
            max_new_tokens=10,
            context_length=16
        )
        assert isinstance(text, str)
        assert text.startswith("First")

    def test_generate_tokens_batch_dimension(self, tiny_gpt_model):
        # Batch size 2
        prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        out = generate_tokens(tiny_gpt_model, prompt, max_new_tokens=5, context_length=16)
        assert out.shape == (2, 7)
        assert (out[:, :2] == prompt).all()

    def test_prompt_exceeding_context_length_gracefully_slides(self, tiny_gpt_model, char_tokenizer):
        long_prompt = "A" * 20  # longer than context_length=16
        out = generate_text_from_prompt(tiny_gpt_model, char_tokenizer, prompt=long_prompt, max_new_tokens=4, context_length=16)
        assert out.startswith(long_prompt)
        assert len(out) == 20 + 4


class TestSamplingStrategies:
    def test_sample_next_token_greedy(self):
        logits = torch.tensor([[1.0, 5.0, 2.0, -1.0]])
        # With temperature=0.0, must pick index 1 (highest logit 5.0)
        token = sample_next_token(logits, temperature=0.0)
        assert token.item() == 1

    def test_sample_top_k_bounds(self):
        logits = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        # With top_k=2, only indices 2 and 3 can ever be sampled
        for _ in range(20):
            token = sample_next_token(logits, temperature=1.0, top_k=2)
            assert token.item() in (2, 3)

    def test_sample_top_p_nucleus_filtering(self):
        # Logits where index 0 has 90%+ probability mass
        logits = torch.tensor([[10.0, 2.0, 0.0, -5.0]])
        for _ in range(15):
            token = sample_next_token(logits, temperature=1.0, top_p=0.5)
            # Only index 0 should pass the 50% cumulative probability cutoff
            assert token.item() == 0

    def test_top_k_larger_than_vocab_handled_safely(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        # Vocab is 3, top_k is 50 -> should clamp and still sample valid token
        token = sample_next_token(logits, temperature=1.0, top_k=50)
        assert token.item() in (0, 1, 2)

    def test_generate_with_strategy_modes(self, tiny_gpt_model, char_tokenizer):
        g_out = generate_with_strategy(tiny_gpt_model, char_tokenizer, "All", max_new_tokens=5, temperature=0.0)
        t_out = generate_with_strategy(tiny_gpt_model, char_tokenizer, "All", max_new_tokens=5, temperature=0.8, top_k=5)
        p_out = generate_with_strategy(tiny_gpt_model, char_tokenizer, "All", max_new_tokens=5, temperature=0.8, top_p=0.8)

        assert g_out.startswith("All")
        assert t_out.startswith("All")
        assert p_out.startswith("All")


class TestScratchSampling:
    def test_numpy_temperature_sample(self):
        logits = np.array([10.0, 0.0, 0.0])
        # Very low temperature should pick index 0 almost deterministically
        sample = sample_temperature_np(logits, temperature=0.01)
        assert sample == 0

    def test_numpy_top_k_sample(self):
        logits = np.array([1.0, 2.0, 8.0, 9.0])
        for _ in range(15):
            sample = sample_top_k_np(logits, k=2, temperature=1.0)
            assert sample in (2, 3)

    def test_numpy_top_p_sample(self):
        logits = np.array([10.0, 1.0, 0.0])
        sample = sample_top_p_np(logits, p=0.8)
        assert sample == 0

    def test_scratch_greedy_generation(self):
        from scratch.gpt_model import MiniGPTScratch
        m = MiniGPTScratch(vocab_size=20, context_length=8, embed_dim=16, num_heads=2, num_layers=1)
        prompt = torch.tensor([[1, 2, 3]])
        out = generate_greedy_scratch(m, prompt, max_new_tokens=5, context_length=8)
        assert out.shape == (1, 8)

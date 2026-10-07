"""
Shared pytest fixtures for Day 115: Preference Optimization & RLHF test suite.
"""
import sys
from pathlib import Path

# Ensure Day 115 directory is in sys.path
DAY_115_ROOT = Path(__file__).resolve().parent.parent
if str(DAY_115_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY_115_ROOT))

import pytest
import torch

from app.config import ModelConfig, RewardModelConfig, DPOTrainingConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.models.minigpt import MiniGPTBackbone
from app.models.reward_model import RewardModel
from app.models.dpo_model import MiniGPTChat, DPOModel
from app.data.collator import PreferenceCollator


@pytest.fixture
def dummy_tokenizer():
    """Lightweight test tokenizer with standard special tokens."""
    vocab = {
        "<|pad|>": 0, "<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|end|>": 4, "<|unk|>": 5,
        "a": 6, "b": 7, "c": 8, "d": 9, "e": 10, " ": 11, "p": 12, "y": 13, "t": 14, "h": 15, "o": 16, "n": 17,
        "1": 18, "2": 19, "3": 20, "4": 21, "+": 22, "=": 23, "?": 24, "!": 25, ".": 26, "-": 27
    }
    tok = ChatTokenizer()
    tok.special_tokens = ["<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|unk|>"]
    tok.itos = {v: k for k, v in vocab.items()}
    tok.stoi = vocab
    return tok


@pytest.fixture
def small_model_config(dummy_tokenizer):
    return ModelConfig(
        vocab_size=dummy_tokenizer.vocab_size,
        context_length=32,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        tie_weights=True
    )


@pytest.fixture
def dummy_mini_chat(small_model_config):
    torch.manual_seed(42)
    return MiniGPTChat(small_model_config)


@pytest.fixture
def dummy_reward_model(small_model_config):
    torch.manual_seed(42)
    rm_cfg = RewardModelConfig(pooling_method="last", dropout=0.0)
    return RewardModel(model_config=small_model_config, reward_config=rm_cfg)


@pytest.fixture
def dummy_dpo_model(dummy_mini_chat):
    torch.manual_seed(42)
    return DPOModel(policy_model=dummy_mini_chat)


@pytest.fixture
def sample_preference_pair():
    return {
        "prompt": "What is Python?",
        "chosen": "Python is a programming language.",
        "rejected": "Python is a snake only.",
        "category": "correctness"
    }


@pytest.fixture
def sample_preference_batch():
    return [
        {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
            "category": "correctness"
        },
        {
            "prompt": "Explain lists.",
            "chosen": "Lists are mutable sequences.",
            "rejected": "Lists are databases in RAM.",
            "category": "conciseness"
        }
    ]

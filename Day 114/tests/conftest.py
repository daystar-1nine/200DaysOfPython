"""
Shared pytest fixtures and configuration for Day 114 test suite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from app.config import ChatModelConfig, SFTTrainingConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.model.minigpt_chat import MiniGPTChat


@pytest.fixture
def tokenizer():
    return ChatTokenizer()


@pytest.fixture
def tiny_config():
    return ChatModelConfig(
        name="TestMiniGPT",
        vocab_size=104,
        context_length=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        tie_weights=True
    )


@pytest.fixture
def tiny_model(tiny_config):
    return MiniGPTChat(tiny_config)


@pytest.fixture
def sample_conversations():
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language."}
            ],
            "category": "python"
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is overfitting?"},
                {"role": "assistant", "content": "Overfitting is when a model learns training noise."}
            ],
            "category": "data_science"
        }
    ]

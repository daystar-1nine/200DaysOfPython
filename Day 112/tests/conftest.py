"""
Shared pytest fixtures and test configuration for Day 112 test suite.
"""
import sys
from pathlib import Path

# Add Day 112 root to sys.path
DAY_112_DIR = Path(__file__).resolve().parent.parent
if str(DAY_112_DIR) not in sys.path:
    sys.path.insert(0, str(DAY_112_DIR))

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

from app.tokenizer.char_tokenizer import CharacterTokenizer
from app.model.gpt import MiniGPT


@pytest.fixture
def sample_text():
    return (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n"
    )


@pytest.fixture
def char_tokenizer(sample_text):
    return CharacterTokenizer.from_text(sample_text)


@pytest.fixture
def tiny_gpt_model():
    # Compact model for rapid test execution
    vocab_size = 32
    context_length = 16
    embed_dim = 32
    num_heads = 2
    num_layers = 2
    return MiniGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0
    )


@pytest.fixture
def synthetic_token_data():
    torch.manual_seed(42)
    return torch.randint(0, 30, (500,), dtype=torch.long)

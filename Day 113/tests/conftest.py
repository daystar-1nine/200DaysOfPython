import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from app.config import ModelConfig, TrainingConfig
from app.model.minigpt import ScalableMiniGPT


@pytest.fixture
def sample_corpus():
    return (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n\n"
        "First Citizen:\n"
        "You are all resolved rather to die than to famish?\n\n"
        "All:\n"
        "Resolved. resolved.\n"
    )


@pytest.fixture
def sample_documents():
    return [
        "Machine learning models require clean, high quality pretraining datasets to avoid degeneration.",
        "Natural language processing has advanced dramatically since the introduction of the Transformer.",
        "Deep learning architectures scale predictably with parameter count, training tokens, and compute.",
        "Short.",  # Low quality
        "https://spam.example.com/buy-now <script>alert('spam')</script> aaaaaa!!!!!!",  # Toxic / spam
    ]


@pytest.fixture
def tiny_config():
    return ModelConfig(
        name="TestTiny",
        vocab_size=32,
        context_length=16,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        tie_weights=True
    )


@pytest.fixture
def tiny_model(tiny_config):
    return ScalableMiniGPT(tiny_config)

"""
Pytest fixtures and configuration for Day 111 test suite.
"""
import sys
from pathlib import Path

# Add Day 111 directory to sys.path
DAY_111_DIR = Path(__file__).resolve().parent.parent
if str(DAY_111_DIR) not in sys.path:
    sys.path.insert(0, str(DAY_111_DIR))

import pytest
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from app.models.bert_classifier import BertSpamClassifier


@pytest.fixture(scope="session")
def tokenizer():
    """Returns a BertTokenizer loaded from prajjwal1/bert-tiny."""
    return BertTokenizer.from_pretrained("prajjwal1/bert-tiny")


@pytest.fixture
def sample_raw_df():
    """Returns a dummy DataFrame resembling raw SMS data."""
    return pd.DataFrame({
        "v1": ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "ham"],
        "v2": [
            "Hey how are you doing today?",
            "WINNER!! You have won a £1,000 cash prize! Text CLAIM to 88888 now!",
            "Can we meet at 3pm tomorrow?",
            "URGENT! Your mobile account has won a bonus. Call 08000930705.",
            "I'm on my way home right now.",
            "Free ringtones! Text YES to 12345 to download.",
            "Don't forget to buy milk on the way back.",
            "Sounds good, see you later!"
        ]
    })


@pytest.fixture
def sample_clean_df():
    """Returns a cleaned DataFrame with standardized columns."""
    return pd.DataFrame({
        "text": [
            "Hey how are you doing today?",
            "WINNER!! You have won a prize text CLAIM to 88888 now",
            "Can we meet at 3pm tomorrow?",
            "URGENT Your mobile account has won a bonus Call 08000930705",
            "I am on my way home right now",
            "Free ringtones Text YES to 12345 to download",
            "Do not forget to buy milk on the way back",
            "Sounds good see you later"
        ],
        "label": [0, 1, 0, 1, 0, 1, 0, 0]
    })


@pytest.fixture
def dummy_dataloader():
    """Returns a small synthetic DataLoader for testing trainer loops."""
    batch_size = 4
    seq_len = 16
    n_samples = 12

    input_ids = torch.randint(100, 3000, (n_samples, seq_len))
    attention_mask = torch.ones((n_samples, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((n_samples, seq_len), dtype=torch.long)
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]])

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

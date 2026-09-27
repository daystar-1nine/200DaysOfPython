import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure Day 108 is on sys.path
day108_dir = Path(__file__).resolve().parent.parent
if str(day108_dir) not in sys.path:
    sys.path.insert(0, str(day108_dir))

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "v1": ["ham", "spam", "ham", "spam", "ham"],
        "v2": [
            "Hey how are you doing today?",
            "WINNER!! Claim your 1000 cash prize now by texting WIN to 88888",
            "Are we still meeting for lunch at 1pm?",
            "URGENT: Your account has been compromised, verify info immediately",
            "Thanks for the update see you soon"
        ]
    })

@pytest.fixture
def sample_clean_df():
    return pd.DataFrame({
        "label": [0, 1, 0, 1, 0],
        "text": [
            "hey how are you doing today",
            "winner claim your 1000 cash prize now by texting win to 88888",
            "are we still meeting for lunch at 1pm",
            "urgent your account has been compromised verify info immediately",
            "thanks for the update see you soon"
        ]
    })

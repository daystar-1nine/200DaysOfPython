import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure Day 109 is on sys.path
day109_dir = Path(__file__).resolve().parent.parent
scratch_dir = day109_dir / "scratch"
for p in [str(day109_dir), str(scratch_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "v1": ["ham", "spam", "ham", "spam", "ham"],
        "v2": [
            "Are you coming to the presentation tonight?",
            "WINNER!! Claim 500 cash now by replying WIN",
            "Can you pick up groceries on your way back?",
            "URGENT: Your parcel delivery is pending, verify immediately",
            "Sounds great, let's catch up later today"
        ]
    })

@pytest.fixture
def sample_clean_df():
    return pd.DataFrame({
        "label": [0, 1, 0, 1, 0],
        "text": [
            "are you coming to the presentation tonight",
            "winner claim 500 cash now by replying win",
            "can you pick up groceries on your way back",
            "urgent your parcel delivery is pending verify immediately",
            "sounds great lets catch up later today"
        ]
    })

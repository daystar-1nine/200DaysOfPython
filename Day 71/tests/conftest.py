"""
Pytest configuration and shared fixtures for Day 71 test suite.
"""

import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
app_dir = base_dir / "app"
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_experiment_df():
    np.random.seed(42)
    n = 200
    groups = ["Control"] * 100 + ["Treatment"] * 100
    conv_c = np.random.binomial(1, 0.05, 100)
    conv_t = np.random.binomial(1, 0.08, 100)
    convs = np.concatenate([conv_c, conv_t])
    revs = convs * np.random.uniform(20.0, 80.0, n)
    bounces = (1 - convs) * np.random.binomial(1, 0.40, n)
    refunds = convs * np.random.binomial(1, 0.02, n)
    durations = np.random.uniform(30.0, 300.0, n)
    
    return pd.DataFrame({
        "user_id": [f"USR-{i}" for i in range(n)],
        "group": groups,
        "converted": convs,
        "revenue": revs,
        "session_duration": durations,
        "orders": convs,
        "bounce": bounces,
        "refunded": refunds
    })

@pytest.fixture
def continuous_groups():
    np.random.seed(123)
    a = np.random.normal(50.0, 10.0, 40)
    b = np.random.normal(55.0, 12.0, 45)
    return a, b

@pytest.fixture
def paired_data():
    np.random.seed(99)
    before = np.random.normal(70.0, 8.0, 30)
    after = before + np.random.normal(4.0, 2.0, 30)
    return before, after

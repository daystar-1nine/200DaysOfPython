"""
Pytest fixtures for Day 69 Confidence Intervals test suite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

@pytest.fixture
def sample_data():
    # 50 values with known seed
    rng = np.random.default_rng(42)
    return rng.normal(loc=100.0, scale=15.0, size=50)

@pytest.fixture
def small_sample():
    return np.array([22.0, 24.0, 25.0, 28.0, 21.0])

@pytest.fixture
def exact_mean_data():
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0])

@pytest.fixture
def binary_outcomes():
    return {"successes": 64, "total": 100}

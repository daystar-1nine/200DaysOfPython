"""
Pytest configuration and shared fixtures for Day 70 test suite.
"""

import sys
from pathlib import Path

# Add Day 70 base directory and Day 70/app to sys.path
base_dir = Path(__file__).resolve().parent.parent
app_dir = base_dir / "app"
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import pytest
import numpy as np

@pytest.fixture
def sample_numeric_data():
    return [22.4, 24.1, 23.8, 25.2, 21.9, 24.5, 23.0, 26.1, 24.8, 23.6]

@pytest.fixture
def known_normal_sample():
    np.random.seed(123)
    return np.random.normal(loc=100.0, scale=15.0, size=50)

@pytest.fixture
def proportion_data():
    return {"successes": 65, "trials": 500, "p_0": 0.10}

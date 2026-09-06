"""
Pytest configuration and shared fixtures for Day 66.
"""
import sys
from pathlib import Path
import pytest

day66_dir = Path(__file__).resolve().parent.parent
if str(day66_dir) not in sys.path:
    sys.path.insert(0, str(day66_dir))

@pytest.fixture
def fair_die_values():
    return [1, 2, 3, 4, 5, 6]

@pytest.fixture
def fair_die_probs():
    return [1/6] * 6

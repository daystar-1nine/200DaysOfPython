"""
Unit tests for input validator module.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from app.validator import (
        validate_groups,
        validate_binary_vector,
        validate_proportions,
        validate_alpha
    )
except (ImportError, ModuleNotFoundError):
    from validator import (
        validate_groups,
        validate_binary_vector,
        validate_proportions,
        validate_alpha
    )

def test_validate_groups_valid():
    df = pd.DataFrame({"group": ["Control", "Treatment", "Control", "Treatment"]})
    ctrl, trt = validate_groups(df)
    assert len(ctrl) == 2
    assert len(trt) == 2

def test_validate_groups_missing_col():
    df = pd.DataFrame({"variant": ["A", "B"]})
    with pytest.raises(ValueError, match="Missing required group column"):
        validate_groups(df)

def test_validate_groups_insufficient_rows():
    df = pd.DataFrame({"group": ["Control"]})
    with pytest.raises(ValueError):
        validate_groups(df)

def test_validate_groups_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_groups(pd.DataFrame())

def test_validate_binary_vector_valid():
    vec = validate_binary_vector([0, 1, 1, 0, 1])
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 5

def test_validate_binary_vector_invalid():
    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        validate_binary_vector([0, 1, 2, 0])

def test_validate_proportions_valid():
    assert validate_proportions(50, 100) == 0.5

def test_validate_proportions_invalid():
    with pytest.raises(ValueError):
        validate_proportions(120, 100)
    with pytest.raises(ValueError):
        validate_proportions(-5, 100)
    with pytest.raises(ValueError):
        validate_proportions(10, 0)

def test_validate_alpha_valid():
    assert validate_alpha(0.05) == 0.05

def test_validate_alpha_invalid():
    with pytest.raises(ValueError):
        validate_alpha(0.0)
    with pytest.raises(ValueError):
        validate_alpha(1.0)
    with pytest.raises(ValueError):
        validate_alpha(-0.01)

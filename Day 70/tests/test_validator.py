"""
Unit tests for input validation module.
"""

import pytest
import numpy as np

try:
    from app.validator import (
        validate_numeric_array,
        validate_alpha,
        validate_alternative,
        validate_proportion,
        validate_counts
    )
except (ImportError, ModuleNotFoundError):
    from validator import (
        validate_numeric_array,
        validate_alpha,
        validate_alternative,
        validate_proportion,
        validate_counts
    )

def test_validate_numeric_array_valid():
    arr = validate_numeric_array([1, 2, 3, 4, 5])
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 5

def test_validate_numeric_array_nan_filtering():
    arr = validate_numeric_array([1.0, np.nan, 3.0, None, 5.0], min_len=3)
    assert len(arr) == 3

def test_validate_numeric_array_too_short():
    with pytest.raises(ValueError, match="Sample size must be at least"):
        validate_numeric_array([1.0], min_len=2)

def test_validate_numeric_array_none():
    with pytest.raises(ValueError, match="Data cannot be None"):
        validate_numeric_array(None)

def test_validate_alpha_valid():
    assert validate_alpha(0.05) == 0.05
    assert validate_alpha(0.01) == 0.01

def test_validate_alpha_invalid_range():
    with pytest.raises(ValueError):
        validate_alpha(0.0)
    with pytest.raises(ValueError):
        validate_alpha(1.0)
    with pytest.raises(ValueError):
        validate_alpha(-0.05)
    with pytest.raises(ValueError):
        validate_alpha(1.2)

def test_validate_alpha_invalid_type():
    with pytest.raises(TypeError):
        validate_alpha("0.05")

def test_validate_alternative_valid():
    assert validate_alternative("two-sided") == "two-sided"
    assert validate_alternative("GREATER") == "greater"
    assert validate_alternative("  less  ") == "less"

def test_validate_alternative_invalid():
    with pytest.raises(ValueError, match="Alternative hypothesis must be one of"):
        validate_alternative("unequal")

def test_validate_proportion_valid():
    assert validate_proportion(0.15) == 0.15

def test_validate_proportion_invalid():
    with pytest.raises(ValueError):
        validate_proportion(0.0)
    with pytest.raises(ValueError):
        validate_proportion(1.0)
    with pytest.raises(TypeError):
        validate_proportion("0.5")

def test_validate_counts_valid():
    succ, tri = validate_counts(25, 100)
    assert succ == 25
    assert tri == 100

def test_validate_counts_invalid():
    with pytest.raises(ValueError, match="cannot exceed total trials"):
        validate_counts(105, 100)
    with pytest.raises(ValueError):
        validate_counts(-1, 100)
    with pytest.raises(ValueError):
        validate_counts(10, 0)

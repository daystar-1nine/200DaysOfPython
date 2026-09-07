"""
Tests for validator module.
"""
import pytest
import numpy as np
from app.validator import (
    validate_numeric_array,
    validate_confidence_level,
    validate_sample_size,
    validate_proportion_counts,
    validate_margin_of_error
)

def test_validate_numeric_array_valid():
    arr = validate_numeric_array([1, 2, 3.5, 4])
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 4

def test_validate_numeric_array_empty_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_numeric_array([])

def test_validate_numeric_array_none_error():
    with pytest.raises(ValueError, match="cannot be None"):
        validate_numeric_array(None)

def test_validate_numeric_array_nan_error():
    with pytest.raises(ValueError, match="contains NaN"):
        validate_numeric_array([1.0, np.nan, 3.0])

def test_validate_numeric_array_inf_error():
    with pytest.raises(ValueError, match="contains Infinite"):
        validate_numeric_array([1.0, np.inf, 3.0])

def test_validate_confidence_level_valid():
    assert validate_confidence_level(0.95) == 0.95
    assert validate_confidence_level(0.90) == 0.90

def test_validate_confidence_level_out_of_bounds():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_confidence_level(1.05)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_confidence_level(0.0)

def test_validate_sample_size_valid():
    assert validate_sample_size(50) == 50

def test_validate_sample_size_zero_or_negative():
    with pytest.raises(ValueError, match="strictly positive"):
        validate_sample_size(0)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_sample_size(-5)

def test_validate_proportion_counts_valid():
    x, n = validate_proportion_counts(40, 100)
    assert x == 40 and n == 100

def test_validate_proportion_counts_exceed_total():
    with pytest.raises(ValueError, match="cannot exceed total"):
        validate_proportion_counts(120, 100)

def test_validate_margin_of_error_positive():
    assert validate_margin_of_error(5.0) == 5.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_margin_of_error(0.0)

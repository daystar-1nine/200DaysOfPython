"""Tests for validator module."""
import numpy as np
import pandas as pd
from app.validator import validate_and_clean_series

def test_validator_normal(sample_odd_data):
    cleaned, meta = validate_and_clean_series(sample_odd_data)
    assert len(cleaned) == 5
    assert not meta["is_empty"]
    assert not meta["is_constant"]
    assert meta["nan_count"] == 0

def test_validator_nan_and_inf(nan_inf_data):
    cleaned, meta = validate_and_clean_series(nan_inf_data)
    assert len(cleaned) == 4
    assert meta["nan_count"] == 1
    assert meta["inf_count"] == 2
    assert list(cleaned) == [10.0, 20.0, 30.0, 40.0]

def test_validator_constant(constant_data):
    cleaned, meta = validate_and_clean_series(constant_data)
    assert meta["is_constant"] is True
    assert len(cleaned) == 4

def test_validator_empty(empty_data):
    cleaned, meta = validate_and_clean_series(empty_data)
    assert meta["is_empty"] is True
    assert len(cleaned) == 0

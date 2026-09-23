"""
Tests for Data Loader, Validator, and Splitter.
"""

import pandas as pd
import pytest
from app.data.validator import DataValidator
from app.data.splitter import DataSplitter


def test_data_validator_valid():
    df = pd.DataFrame({
        "label": ["ham", "spam", "ham"],
        "text": ["hello friend", "free lottery winner", "call me back"]
    })
    is_valid, errors = DataValidator.validate(df)
    assert is_valid is True
    assert len(errors) == 0


def test_data_validator_missing_column():
    df = pd.DataFrame({
        "message": ["hello friend", "free lottery winner"]
    })
    is_valid, errors = DataValidator.validate(df)
    assert is_valid is False
    assert any("must contain 'text' and 'label'" in e for e in errors)


def test_data_validator_invalid_labels():
    df = pd.DataFrame({
        "label": ["ham", "unknown_label"],
        "text": ["hello friend", "free lottery winner"]
    })
    is_valid, errors = DataValidator.validate(df)
    assert is_valid is False
    assert any("Found invalid labels" in e for e in errors)


def test_dataset_splitter_ratios():
    data = {
        "label": ["ham"] * 80 + ["spam"] * 20,
        "text": [f"Message text {i}" for i in range(100)]
    }
    df = pd.DataFrame(data)
    train, val, test = DataSplitter.split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    
    # Check stratified spam presence
    assert (train["label"] == "spam").sum() > 0
    assert (val["label"] == "spam").sum() > 0
    assert (test["label"] == "spam").sum() > 0

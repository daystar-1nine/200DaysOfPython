import pytest
import pandas as pd
from app.data.cleaner import clean_sms_data
from app.data.validator import validate_sms_data
from app.data.splitter import split_dataset

def test_clean_sms_data_mapping(sample_raw_df):
    clean_df = clean_sms_data(sample_raw_df)
    assert set(clean_df["label"].unique()) == {0, 1}
    assert clean_df["label"].dtype == int
    assert len(clean_df) == len(sample_raw_df)

def test_clean_sms_data_drops_duplicates():
    df_with_dups = pd.DataFrame({
        "label": ["ham", "ham", "spam"],
        "text": ["Duplicate message", "Duplicate message", "Spam message"]
    })
    clean_df = clean_sms_data(df_with_dups, drop_duplicates=True)
    assert len(clean_df) == 2
    assert "Duplicate message" in clean_df["text"].values

def test_clean_sms_data_drops_na():
    df_na = pd.DataFrame({
        "label": ["ham", None, "spam"],
        "text": ["hello", "world", None]
    })
    clean_df = clean_sms_data(df_na)
    assert len(clean_df) == 1
    assert clean_df["text"].iloc[0] == "hello"

def test_validate_sms_data_valid(sample_clean_df):
    assert validate_sms_data(sample_clean_df) is True

def test_validate_sms_data_missing_columns():
    df_bad = pd.DataFrame({"colA": [1], "colB": [2]})
    with pytest.raises(ValueError, match="DataFrame must contain"):
        validate_sms_data(df_bad)

def test_validate_sms_data_single_class():
    df_single = pd.DataFrame({"label": [0, 0, 0], "text": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="both positive"):
        validate_sms_data(df_single)

def test_split_dataset_stratification():
    df = pd.DataFrame({
        "label": [0]*80 + [1]*20,
        "text": [f"msg {i}" for i in range(100)]
    })
    train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    # Check proportion of spam is 20% in train
    assert train_df["label"].sum() == 14

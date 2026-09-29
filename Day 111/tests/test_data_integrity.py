"""
Unit tests for data loading, cleaning, deduplication, and stratification.
"""
from pathlib import Path
import pytest
import pandas as pd
from app.data.loader import load_raw_sms_data
from app.data.cleaner import clean_sms_data
from app.data.splitter import split_dataset


class TestDataLoading:
    def test_load_nonexistent_file_raises_error(self, tmp_path):
        bad_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_raw_sms_data(bad_path)

    def test_load_valid_csv(self, tmp_path):
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("v1,v2\nham,Hello\nspam,Win now\n", encoding="utf-8")
        df = load_raw_sms_data(csv_file)
        assert len(df) == 2
        assert list(df.columns) == ["v1", "v2"]

    def test_load_different_encodings(self, tmp_path):
        csv_file = tmp_path / "latin1_data.csv"
        csv_file.write_bytes("v1,v2\nham,Café\nspam,£100 prize\n".encode("latin-1"))
        df = load_raw_sms_data(csv_file)
        assert len(df) == 2


class TestDataCleaning:
    def test_rename_v1_v2_columns(self, sample_raw_df):
        cleaned = clean_sms_data(sample_raw_df)
        assert "text" in cleaned.columns
        assert "label" in cleaned.columns

    def test_label_mapping_binary(self, sample_raw_df):
        cleaned = clean_sms_data(sample_raw_df)
        assert set(cleaned["label"].unique()).issubset({0, 1})

    def test_whitespace_stripped(self):
        raw = pd.DataFrame({"v1": ["ham", "spam"], "v2": ["  hello world  ", "  call now!  "]})
        cleaned = clean_sms_data(raw)
        assert cleaned.iloc[0]["text"] == "hello world"
        assert cleaned.iloc[1]["text"] == "call now!"

    def test_drop_missing_values(self):
        raw = pd.DataFrame({"v1": ["ham", None, "spam"], "v2": ["valid text", "missing label", None]})
        cleaned = clean_sms_data(raw)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["text"] == "valid text"

    def test_drop_empty_strings(self):
        raw = pd.DataFrame({"v1": ["ham", "spam"], "v2": ["", "   "]})
        cleaned = clean_sms_data(raw)
        assert len(cleaned) == 0

    def test_drop_duplicate_texts(self):
        raw = pd.DataFrame({
            "v1": ["ham", "ham", "spam"],
            "v2": ["Exact duplicate text", "Exact duplicate text", "Unique spam text"]
        })
        cleaned = clean_sms_data(raw)
        assert len(cleaned) == 2

    def test_category_message_columns_handling(self):
        raw = pd.DataFrame({"Category": ["ham", "spam"], "Message": ["hi there", "claim award"]})
        cleaned = clean_sms_data(raw)
        assert "text" in cleaned.columns
        assert "label" in cleaned.columns
        assert cleaned.iloc[0]["label"] == 0
        assert cleaned.iloc[1]["label"] == 1


class TestDataSplitting:
    def test_split_ratios_sum_to_one(self, sample_clean_df):
        with pytest.raises(AssertionError):
            split_dataset(sample_clean_df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

    def test_splits_non_overlapping(self, sample_clean_df):
        train_df, val_df, test_df = split_dataset(sample_clean_df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        train_texts = set(train_df["text"])
        val_texts = set(val_df["text"])
        test_texts = set(test_df["text"])

        assert len(train_texts.intersection(val_texts)) == 0
        assert len(train_texts.intersection(test_texts)) == 0
        assert len(val_texts.intersection(test_texts)) == 0

    def test_total_samples_preserved(self, sample_clean_df):
        train_df, val_df, test_df = split_dataset(sample_clean_df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        assert len(train_df) + len(val_df) + len(test_df) == len(sample_clean_df)

    def test_stratification_retains_both_classes(self, sample_clean_df):
        train_df, val_df, test_df = split_dataset(sample_clean_df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        assert 0 in train_df["label"].values and 1 in train_df["label"].values
        assert 0 in val_df["label"].values and 1 in val_df["label"].values
        assert 0 in test_df["label"].values and 1 in test_df["label"].values

    def test_deterministic_split_with_seed(self, sample_clean_df):
        t1, v1, te1 = split_dataset(sample_clean_df, random_seed=42)
        t2, v2, te2 = split_dataset(sample_clean_df, random_seed=42)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(te1, te2)

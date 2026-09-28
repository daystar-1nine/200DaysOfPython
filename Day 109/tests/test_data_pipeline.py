import pytest
import pandas as pd
import numpy as np
from app.data.loader import load_raw_data
from app.data.cleaner import clean_sms_data
from app.data.validator import validate_sms_data
from app.data.splitter import split_dataset
from app.preprocessing.encoder import encode_tokens, encode_texts
from app.preprocessing.padding import pad_sequences
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.tokenizer import tokenize_text
from app.config import PAD_IDX, UNK_IDX

def test_load_raw_data_success(tmp_path):
    f = tmp_path / "sample.csv"
    f.write_text("v1,v2\nham,Meeting at 3pm\nspam,Free holiday voucher\n", encoding="utf-8")
    df = load_raw_data(f)
    assert len(df) == 2
    assert list(df.columns) == ["label", "text"]

def test_clean_sms_data_mapping_and_types(sample_raw_df):
    clean_df = clean_sms_data(sample_raw_df)
    assert set(clean_df["label"].unique()) == {0, 1}
    assert clean_df["label"].dtype == int
    assert len(clean_df) == len(sample_raw_df)

def test_clean_sms_data_deduplication():
    df_dup = pd.DataFrame({
        "label": [0, 0, 1],
        "text": ["identical message", "identical message", "urgent spam"]
    })
    cleaned = clean_sms_data(df_dup, drop_duplicates=True)
    assert len(cleaned) == 2

def test_validate_sms_data_valid(sample_clean_df):
    assert validate_sms_data(sample_clean_df) is True

def test_validate_sms_data_missing_column():
    bad_df = pd.DataFrame({"random_col": [1, 2]})
    with pytest.raises(ValueError, match="DataFrame must contain"):
        validate_sms_data(bad_df)

def test_validate_sms_data_empty():
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_sms_data(pd.DataFrame({"label": [], "text": []}))

def test_split_dataset_proportions():
    df = pd.DataFrame({
        "label": [0]*70 + [1]*30,
        "text": [f"message {i}" for i in range(100)]
    })
    train_df, val_df, test_df = split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    # Check stratification preservation
    assert train_df["label"].sum() == 21
    assert val_df["label"].sum() == 5 or val_df["label"].sum() == 4

def test_encode_tokens_unknowns():
    v = Vocabulary()
    v.fit([["urgent", "call"]])
    tokens = ["urgent", "unknown_word", "call"]
    encoded = encode_tokens(tokens, v)
    assert encoded[0] == v.get_idx("urgent")
    assert encoded[1] == UNK_IDX
    assert encoded[2] == v.get_idx("call")

def test_encode_texts_batch():
    v = Vocabulary()
    v.fit([["hello", "friend"]])
    texts = ["hello friend", "mystery text"]
    encoded = encode_texts(texts, v, tokenize_text)
    assert len(encoded) == 2
    assert encoded[0] == [v.get_idx("hello"), v.get_idx("friend")]
    assert encoded[1] == [UNK_IDX, UNK_IDX]

@pytest.mark.parametrize("max_len", [5, 10, 15, 25])
def test_pad_sequences_batch_lengths(max_len):
    batch = [[1, 2], [3, 4, 5, 6, 7, 8, 9, 10]]
    padded = pad_sequences(batch, max_len=max_len)
    assert padded.shape == (2, max_len)
    assert isinstance(padded, np.ndarray)

@pytest.mark.parametrize("text, expected_tokens", [
    ("hello world", ["hello", "world"]),
    ("win $500 now", ["win", "currency", "500", "now"]),
    ("urgent: call!", ["urgent", "call"])
])
def test_tokenize_text_examples(text, expected_tokens):
    assert tokenize_text(text) == expected_tokens

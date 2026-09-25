"""
Tests ensuring Zero Data Leakage in Day 106: RNNs & Sequential Text Learning.
"""

import pandas as pd
from app.preprocessing.vocabulary import Vocabulary
from app.data.splitter import DataSplitter


def test_leakage_test_tokens_not_in_train_vocabulary():
    train_texts = [
        "win cash prize",
        "urgent lottery winner",
        "free reward claim"
    ]
    test_texts = [
        "exclusive_cryptocurrency_token_xyz"
    ]

    vocab = Vocabulary(min_freq=1).fit(train_texts)

    # The test-only word MUST NOT be present in train vocabulary
    assert "exclusive_cryptocurrency_token_xyz" not in vocab.word2idx
    # Encoding it must resolve to UNK
    assert vocab.token_to_id("exclusive_cryptocurrency_token_xyz") == vocab.UNK_ID


def test_data_splitter_no_index_overlap():
    df = pd.DataFrame({
        "label": ["ham"] * 80 + ["spam"] * 20,
        "text": [f"Message text {i}" for i in range(100)],
        "target": [0] * 80 + [1] * 20
    })

    train_df, val_df, test_df = DataSplitter.split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    # Ensure mutually exclusive partition sizes
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check texts in subsets do not overlap
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    assert len(train_texts.intersection(val_texts)) == 0
    assert len(train_texts.intersection(test_texts)) == 0
    assert len(val_texts.intersection(test_texts)) == 0


def test_data_splitter_stratification_preserves_class_ratio():
    df = pd.DataFrame({
        "label": ["ham"] * 80 + ["spam"] * 20,
        "text": [f"Message {i}" for i in range(100)],
        "target": [0] * 80 + [1] * 20
    })

    train_df, val_df, test_df = DataSplitter.split(df, 0.70, 0.15, 0.15, random_seed=42)

    # 20% spam overall
    train_spam_rate = train_df["target"].mean()
    val_spam_rate = val_df["target"].mean()
    test_spam_rate = test_df["target"].mean()

    assert abs(train_spam_rate - 0.20) <= 0.02
    assert abs(val_spam_rate - 0.20) <= 0.02
    assert abs(test_spam_rate - 0.20) <= 0.02


def test_vocabulary_never_fitted_on_test_data():
    corpus = pd.DataFrame({
        "text": ["train word one", "train word two", "unseen test token"],
        "split": ["train", "train", "test"]
    })

    train_only = corpus[corpus["split"] == "train"]["text"]
    vocab = Vocabulary(min_freq=1).fit(train_only)

    assert "unseen" not in vocab.word2idx
    assert "token" not in vocab.word2idx

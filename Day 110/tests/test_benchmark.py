"""
Unit tests for Data Pipeline, Benchmarks, Evaluation, and Analysis.
"""
import pytest
import numpy as np
import pandas as pd
from app.data.cleaner import clean_sms_data
from app.data.validator import validate_sms_data
from app.data.splitter import split_dataset
from app.preprocessing.tokenizer import tokenize_text
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.padding import pad_sequences
from app.evaluation.metrics import compute_metrics
from app.evaluation.confusion import compute_confusion_matrix
from app.evaluation.threshold import analyze_thresholds
from app.analysis.attention_analysis import compute_head_entropy, compute_head_correlation
from app.analysis.error_analysis import perform_error_analysis


def test_clean_sms_data_and_validator():
    raw_df = pd.DataFrame({
        "v1": ["ham", "spam", "ham", "ham", "spam"],
        "v2": ["Hello world", "Win free prize", "See you later", "Hello world", "Call now"]
    })
    cleaned = clean_sms_data(raw_df)
    assert len(cleaned) == 4  # 1 duplicate dropped
    stats = validate_sms_data(cleaned)
    assert stats["is_valid"] is True
    assert stats["total_samples"] == 4


def test_stratified_split_proportions():
    df = pd.DataFrame({
        "text": [f"msg {i}" for i in range(100)],
        "label": [0] * 70 + [1] * 30
    })
    tr, va, te = split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert len(tr) in [69, 70]
    assert len(va) in [15, 16]
    assert len(te) == 15
    assert len(tr) + len(va) + len(te) == 100
    # Stratification checks
    assert tr["label"].sum() in [20, 21]
    assert va["label"].sum() in [4, 5]
    assert te["label"].sum() in [4, 5]


def test_zero_data_leakage_strict():
    train_tokens = [["urgent", "call", "now"], ["free", "entry", "claim"]]
    test_tokens = [["urgent", "secret", "alien"]]  # 'secret' and 'alien' are unseen

    vocab = Vocabulary(min_freq=1)
    vocab.fit(train_tokens)

    encoded_test = vocab.encode(test_tokens[0])
    assert encoded_test[0] == vocab.token2idx["urgent"]
    assert encoded_test[1] == vocab.unk_idx
    assert encoded_test[2] == vocab.unk_idx


def test_metrics_computation():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_probs)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_confusion_matrix_values():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8])  # 1 FP, 1 FN
    cm, counts = compute_confusion_matrix(y_true, y_probs)
    assert counts["TN"] == 1
    assert counts["FP"] == 1
    assert counts["FN"] == 1
    assert counts["TP"] == 1


def test_threshold_analysis():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.2, 0.3, 0.7, 0.8])
    df, best_th, best_m = analyze_thresholds(y_true, y_probs)
    assert len(df) == 19
    assert best_m["f1"] == 1.0


def test_head_entropy_and_correlation():
    # 2 heads, seq_len 4
    weights = np.zeros((2, 4, 4))
    # Head 0: uniform
    weights[0] = 0.25
    # Head 1: peaked on diagonal
    weights[1] = np.eye(4)

    entropy = compute_head_entropy(weights)
    assert entropy.shape == (2, 4)
    # Uniform has higher entropy than peaked
    assert (entropy[0] > entropy[1]).all()

    corr = compute_head_correlation(weights)
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)


def test_error_analysis():
    texts = ["win cash", "hello mom"]
    y_true = np.array([1, 0])
    y_probs = np.array([0.9, 0.1])
    err_df = perform_error_analysis(texts, y_true, y_probs)
    assert len(err_df) == 2
    assert err_df["is_error"].sum() == 0

import pytest
import numpy as np
import pandas as pd
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.tokenizer import tokenize_text
from app.preprocessing.padding import pad_sequence, pad_sequences
from app.preprocessing.encoder import encode_tokens, encode_texts
from app.config import PAD_IDX, UNK_IDX, PAD_TOKEN, UNK_TOKEN
from app.analysis.attention_analysis import extract_top_attention_tokens, aggregate_token_attention
from app.analysis.error_analysis import perform_error_analysis

def test_zero_data_leakage_strict():
    train_corpus = [["hello", "world", "claim"], ["win", "cash"]]
    test_corpus = [["unseen_secret_token_999", "world"]]
    
    vocab = Vocabulary(max_size=100)
    vocab.fit(train_corpus)
    
    # Assert unseen test token is not in train vocab
    assert "unseen_secret_token_999" not in vocab.word2idx
    assert vocab.get_idx("unseen_secret_token_999") == UNK_IDX
    assert vocab.get_idx("hello") >= 2

def test_tokenize_text_preserves_signals():
    text = "Win £1000 cash or $500 bonus now!"
    tokens = tokenize_text(text)
    assert "currency" in tokens
    assert "win" in tokens
    assert "bonus" in tokens

def test_padding_and_truncation():
    seq = [1, 2, 3, 4, 5]
    padded = pad_sequence(seq, max_len=8, pad_value=PAD_IDX, padding="post")
    assert padded == [1, 2, 3, 4, 5, 0, 0, 0]
    
    truncated = pad_sequence(seq, max_len=3, truncating="post")
    assert truncated == [1, 2, 3]

def test_extract_top_attention_tokens():
    tokens = ["free", "entry", "win", "cash", "today"]
    weights = np.array([0.45, 0.05, 0.35, 0.10, 0.05])
    top3 = extract_top_attention_tokens(tokens, weights, k=3)
    
    assert len(top3) == 3
    assert top3[0] == ("free", 0.45)
    assert top3[1] == ("win", 0.35)
    assert top3[2] == ("cash", 0.10)

def test_aggregate_token_attention():
    texts = [["win", "prize"], ["win", "cash"]]
    weights = np.array([
        [0.6, 0.4],
        [0.8, 0.2]
    ])
    df = aggregate_token_attention(texts, weights)
    assert "token" in df.columns
    assert "average_attention" in df.columns
    # 'win' appears twice with weights 0.6 and 0.8 -> avg 0.7
    win_row = df[df["token"] == "win"]
    assert len(win_row) == 1
    assert np.isclose(win_row["average_attention"].iloc[0], 0.7)

def test_perform_error_analysis_comparisons(tmp_path):
    texts = ["msg 0", "msg 1", "msg 2"]
    tokenized = [["msg", "0"], ["msg", "1"], ["msg", "2"]]
    y_true = np.array([0, 1, 1])
    # GRU gets index 1 wrong (prob=0.3 < 0.5)
    gru_probs = np.array([0.1, 0.3, 0.9])
    # Attention gets index 1 right (prob=0.85 >= 0.5)
    attn_probs = np.array([0.1, 0.85, 0.9])
    attn_weights = np.array([[0.5, 0.5], [0.8, 0.2], [0.6, 0.4]])
    
    res = perform_error_analysis(
        texts=texts,
        tokenized_texts=tokenized,
        y_true=y_true,
        gru_probs=gru_probs,
        attn_probs=attn_probs,
        attn_weights=attn_weights,
        output_dir=tmp_path
    )
    
    summary = res["comparative_summary"]
    # GRU was wrong on index 1, Attention was right
    assert summary["gru_wrong_attn_right_count"] == 1
    assert summary["gru_wrong_attn_right_indices"] == [1]
    assert summary["both_wrong_count"] == 0
    assert (tmp_path / "attention_errors.csv").exists()

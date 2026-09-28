"""
Benchmark suite comparing TF-IDF+LR, SimpleRNN, LSTM, GRU, GRU+Attention, and Mini Transformer.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import pandas as pd
import numpy as np
import torch
from app.models.baselines import RecurrentBaselineClassifier, TfidfBaseline
from app.models.gru_attention import GRUAttentionClassifier
from app.models.classifier import MiniTransformerClassifier
from app.training.trainer import Trainer
from app.evaluation.metrics import compute_metrics


def run_comprehensive_benchmark(
    vocab_size: int,
    max_length: int,
    texts_train: List[str],
    texts_test: List[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 15,
    output_csv: Path = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains each architecture on the exact same dataset splits,
    measures parameter counts and real training time, and records evaluation metrics.
    """
    records: List[Dict[str, Any]] = []
    test_probs_dict: Dict[str, np.ndarray] = {}

    # 1. TF-IDF + Logistic Regression
    t0 = time.time()
    tfidf = TfidfBaseline(max_features=1000)
    tfidf.fit(texts_train, y_train)
    t_tfidf = time.time() - t0
    probs_tfidf = tfidf.predict_proba(texts_test)
    m_tfidf = compute_metrics(y_test, probs_tfidf)
    test_probs_dict["TF-IDF + LR"] = probs_tfidf
    records.append({
        "Model": "TF-IDF + LR",
        "Parameters": tfidf.count_parameters(),
        "Training Time (s)": round(t_tfidf, 2),
        "Accuracy": round(m_tfidf["accuracy"], 4),
        "Precision": round(m_tfidf["precision"], 4),
        "Recall": round(m_tfidf["recall"], 4),
        "F1": round(m_tfidf["f1"], 4),
        "ROC-AUC": round(m_tfidf["roc_auc"], 4)
    })

    # Neural Models Configuration
    neural_models = [
        ("SimpleRNN", lambda: RecurrentBaselineClassifier("rnn", vocab_size=vocab_size, embedding_dim=128, hidden_dim=128)),
        ("LSTM", lambda: RecurrentBaselineClassifier("lstm", vocab_size=vocab_size, embedding_dim=128, hidden_dim=128)),
        ("GRU", lambda: RecurrentBaselineClassifier("gru", vocab_size=vocab_size, embedding_dim=128, hidden_dim=128)),
        ("GRU + Attention", lambda: GRUAttentionClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128, attn_dim=64)),
        ("Mini Transformer", lambda: MiniTransformerClassifier(vocab_size=vocab_size, max_length=max_length, d_model=128, num_heads=4, d_ff=256, num_layers=2))
    ]

    for name, builder in neural_models:
        model = builder()
        trainer = Trainer(model, learning_rate=1e-3)
        history, train_time = trainer.fit(
            x_train, y_train, x_val, y_val,
            epochs=epochs,
            batch_size=32,
            patience=4
        )
        probs = trainer.predict_proba(x_test)
        metrics = compute_metrics(y_test, probs)
        test_probs_dict[name] = probs

        records.append({
            "Model": name,
            "Parameters": model.count_parameters(),
            "Training Time (s)": round(train_time, 2),
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4)
        })

    df = pd.DataFrame(records)
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df, test_probs_dict

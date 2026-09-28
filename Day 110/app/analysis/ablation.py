"""
Ablation study module for Transformer components.
Evaluates the contribution of Positional Encoding, Residual Connections,
Layer Normalization, and Attention Padding Masks.
"""
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
from app.models.classifier import MiniTransformerClassifier
from app.training.trainer import Trainer
from app.evaluation.metrics import compute_metrics


def run_ablation_study(
    vocab_size: int,
    max_length: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 15,
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Executes controlled ablation experiments by toggling one component at a time.
    """
    configs = [
        {
            "name": "Full Transformer",
            "kwargs": dict(use_positional_encoding=True, use_residual=True, use_layer_norm=True, use_padding_mask=True),
            "observation": "Baseline complete architecture with residual connections and layer normalization."
        },
        {
            "name": "No positional encoding",
            "kwargs": dict(use_positional_encoding=False, use_residual=True, use_layer_norm=True, use_padding_mask=True),
            "observation": "Bag-of-words self-attention without token position order awareness."
        },
        {
            "name": "No residual",
            "kwargs": dict(use_positional_encoding=True, use_residual=False, use_layer_norm=True, use_padding_mask=True),
            "observation": "Removes skip-connections; degrades gradient flow and representation preservation."
        },
        {
            "name": "No LayerNorm",
            "kwargs": dict(use_positional_encoding=True, use_residual=True, use_layer_norm=False, use_padding_mask=True),
            "observation": "Without normalization; activations and variance drift across layers."
        },
        {
            "name": "No padding mask",
            "kwargs": dict(use_positional_encoding=True, use_residual=True, use_layer_norm=True, use_padding_mask=False),
            "observation": "Attention attends over meaningless pad tokens, diluting semantic attention distribution."
        },
    ]

    records: List[Dict[str, Any]] = []

    for cfg in configs:
        model = MiniTransformerClassifier(
            vocab_size=vocab_size,
            max_length=max_length,
            d_model=128,
            num_heads=4,
            d_ff=256,
            num_layers=2,
            dropout=0.1,
            **cfg["kwargs"]
        )
        trainer = Trainer(model, learning_rate=1e-3)
        history, train_time = trainer.fit(
            x_train, y_train, x_val, y_val,
            epochs=epochs,
            batch_size=32,
            patience=4
        )

        test_probs = trainer.predict_proba(x_test)
        metrics = compute_metrics(y_test, test_probs, threshold=0.5)

        records.append({
            "Configuration": cfg["name"],
            "Parameters": model.count_parameters(),
            "Training Time (s)": round(train_time, 2),
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "Observation": cfg["observation"]
        })

    df = pd.DataFrame(records)
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df

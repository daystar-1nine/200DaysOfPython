"""
Decision threshold tuning and sensitivity analysis for BERT classification.
"""
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from app.evaluation.metrics import compute_metrics


def sweep_thresholds(
    y_true: Union[np.ndarray, List[int]],
    y_probs: Union[np.ndarray, List[float]],
    thresholds: List[float] = None,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Sweeps probability decision thresholds from 0.05 to 0.95 and computes all evaluation metrics.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)

    records = []
    for t in thresholds:
        preds = (y_probs_arr >= t).astype(int)
        m = compute_metrics(y_true_arr, preds, y_probs_arr)
        records.append({
            "threshold": t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"]
        })

    df = pd.DataFrame(records)

    # Identify optimal threshold maximizing F1
    best_idx = df["f1"].idxmax()
    best_thresh = df.loc[best_idx, "threshold"]
    best_f1 = df.loc[best_idx, "f1"]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Threshold analysis saved to: {output_path} | Best threshold: {best_thresh} (F1: {best_f1:.4f})")

    return df

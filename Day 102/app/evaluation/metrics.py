"""Evaluation metrics and timing utilities."""
import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict[str, float]:
    """Computes comprehensive binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["avg_precision"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = 0.5
            metrics["avg_precision"] = 0.0
    return metrics

def measure_inference_time(model, X_sample, n_iterations: int = 5) -> float:
    """Measures average inference latency in milliseconds."""
    # Warmup
    _ = model.predict(X_sample[:min(5, len(X_sample))])
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times))

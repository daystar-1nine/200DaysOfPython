"""
Classification threshold sweeping and optimal threshold analysis.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app.evaluation.metrics import compute_metrics


def analyze_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 19)
) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Sweeps decision thresholds and determines the threshold maximizing F1 score.
    """
    records: List[Dict[str, float]] = []
    best_f1 = -1.0
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        m = compute_metrics(y_true, y_probs, threshold=float(t))
        records.append(m)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(t)
            best_metrics = m

    df = pd.DataFrame(records)
    return df, best_thresh, best_metrics


def plot_threshold_curves(
    df: pd.DataFrame,
    best_threshold: float,
    save_path: Path = None
) -> plt.Figure:
    """Plots Precision, Recall, and F1 across decision thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", color="royalblue", lw=2)
    ax.plot(df["threshold"], df["recall"], label="Recall", color="darkorange", lw=2)
    ax.plot(df["threshold"], df["f1"], label="F1 Score", color="forestgreen", lw=2.5)
    ax.axvline(best_threshold, color="crimson", linestyle="--", label=f"Optimal Thr={best_threshold:.2f}")

    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Decision Threshold Trade-off Curves", fontweight="bold", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig

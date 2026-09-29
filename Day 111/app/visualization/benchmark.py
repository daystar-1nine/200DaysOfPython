"""
Visualization module for benchmark comparisons, training trajectories, and threshold curves.
"""
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_benchmark_comparison(benchmark_df: pd.DataFrame, output_path: Path) -> None:
    """Plots a multi-metric bar chart comparing all models in the benchmark."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    models = benchmark_df["Model"].tolist()

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2b5c8f", "#41924b", "#e67e22", "#9b59b6", "#e74c3c"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2) * width + width / 2
        values = benchmark_df[metric].values
        ax.bar(x + offset, values, width, label=metric, color=color, alpha=0.85)

    ax.set_title("Cross-Architecture Benchmark Performance on SMS Spam", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Benchmark chart saved to: {output_path}")


def plot_training_curves(
    history_a: Dict[str, List[float]],
    history_b: Dict[str, List[float]],
    output_path: Path
) -> None:
    """Plots training and validation loss & F1 curves for Frozen vs Fine-Tuned BERT."""
    epochs_a = history_a["epoch"]
    epochs_b = history_b["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    ax1.plot(epochs_a, history_a["train_loss"], "o--", color="#3498db", label="Frozen Train Loss")
    ax1.plot(epochs_a, history_a["val_loss"], "s-", color="#2980b9", label="Frozen Val Loss")
    ax1.plot(epochs_b, history_b["train_loss"], "o--", color="#e74c3c", label="Fine-Tuned Train Loss")
    ax1.plot(epochs_b, history_b["val_loss"], "s-", color="#c0392b", label="Fine-Tuned Val Loss")
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("BCE Loss", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # F1 plot
    ax2.plot(epochs_a, history_a["val_f1"], "s-", color="#2980b9", label="Frozen Val F1")
    ax2.plot(epochs_b, history_b["val_f1"], "s-", color="#c0392b", label="Fine-Tuned Val F1")
    ax2.set_title("Validation F1 Score Across Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("F1 Score", fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {output_path}")


def plot_threshold_curve(threshold_df: pd.DataFrame, output_path: Path) -> None:
    """Plots Precision, Recall, and F1 across decision thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#3498db", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#e67e22", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1 Score", color="#27ae60", linewidth=2.5)

    best_idx = threshold_df["f1"].idxmax()
    best_thresh = threshold_df.loc[best_idx, "threshold"]
    best_f1 = threshold_df.loc[best_idx, "f1"]

    ax.axvline(best_thresh, color="#e74c3c", linestyle="--", alpha=0.8, label=f"Best Threshold ({best_thresh:.2f})")
    ax.scatter([best_thresh], [best_f1], color="#e74c3c", s=60, zorder=5)

    ax.set_title("Decision Threshold Sensitivity (Precision vs Recall vs F1)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Probability Threshold", fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Threshold curve saved to: {output_path}")

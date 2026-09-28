"""
Benchmarking, experiment sweeps, and training visualization plots.
"""
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: Dict[str, List[float]], save_path: Path = None) -> plt.Figure:
    """Plots training and validation loss & accuracy side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="royalblue")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontweight="bold", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="royalblue")
    ax2.plot(epochs, history["val_acc"], "s--", label="Val Acc", color="forestgreen")
    if "val_f1" in history:
        ax2.plot(epochs, history["val_f1"], "^-.", label="Val F1", color="darkorange")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("Training & Validation Metrics", fontweight="bold", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_benchmark_comparison(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Bar chart comparing Accuracy, F1, and ROC-AUC across benchmarked architectures."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["Accuracy"], width, label="Accuracy", color="#3b82f6")
    ax.bar(x, df["F1"], width, label="F1 Score", color="#10b981")
    ax.bar(x + width, df["ROC-AUC"], width, label="ROC-AUC", color="#f59e0b")

    ax.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax.set_title("Comprehensive Architecture Benchmark Comparison", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_parameters_vs_time(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Scatter plot illustrating Model Parameters vs Training Time trade-offs."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for _, row in df.iterrows():
        ax.scatter(row["Parameters"], row["Training Time (s)"], s=180, alpha=0.8, edgecolors="k")
        ax.annotate(
            row["Model"],
            (row["Parameters"], row["Training Time (s)"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9
        )

    ax.set_xlabel("Trainable Parameters", fontweight="bold")
    ax.set_ylabel("Training Time (seconds)", fontweight="bold")
    ax.set_title("Model Complexity vs Training Time", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_ablation_comparison(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """Horizontal bar chart showing the performance impact of removing components."""
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))

    bars = ax.barh(y, df["F1"], color="#6366f1", height=0.6, edgecolor="k")
    ax.set_yticks(y)
    ax.set_yticklabels(df["Configuration"], fontweight="bold")
    ax.set_xlabel("F1 Score", fontweight="bold")
    ax.set_title("Transformer Ablation Study: Impact on F1 Score", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 1.1)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontweight="bold", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_sequence_length_complexity(
    lengths: List[int],
    transformer_times: List[float],
    gru_times: List[float],
    save_path: Path = None
) -> plt.Figure:
    """Plots Training Time vs Sequence Length demonstrating O(n^2) vs O(n) trends."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths, transformer_times, "o-", label="Transformer (Self-Attention O(n²))", color="#ef4444", lw=2.5)
    ax.plot(lengths, gru_times, "s--", label="GRU (Recurrence O(n))", color="#3b82f6", lw=2.5)

    ax.set_xlabel("Input Sequence Length (Tokens)", fontweight="bold")
    ax.set_ylabel("Training Time (seconds)", fontweight="bold")
    ax.set_title("Computational Complexity: Transformer vs GRU over Sequence Length", fontweight="bold", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_experiment_sweep(
    param_name: str,
    param_values: List[Any],
    f1_scores: List[float],
    train_times: List[float],
    save_path: Path = None
) -> plt.Figure:
    """Dual-axis plot showing performance and training time across hyperparameter values."""
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    color_f1 = "#10b981"
    color_time = "#6366f1"

    ax1.set_xlabel(param_name, fontweight="bold")
    ax1.set_ylabel("F1 Score", color=color_f1, fontweight="bold")
    line1 = ax1.plot(param_values, f1_scores, "o-", color=color_f1, lw=2.5, label="F1 Score")
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Time (s)", color=color_time, fontweight="bold")
    line2 = ax2.plot(param_values, train_times, "s--", color=color_time, lw=2.5, label="Training Time")
    ax2.tick_params(axis="y", labelcolor=color_time)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")

    plt.title(f"Hyperparameter Sweep: {param_name}", fontweight="bold", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig

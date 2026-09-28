"""
Confusion matrix generation and visualization.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Computes binary confusion matrix and returns dictionary of TN, FP, FN, TP."""
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return cm, {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Path = None
) -> plt.Figure:
    """Renders formatted confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Ham (0)", "Spam (1)"],
        yticklabels=["Ham (0)", "Spam (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig

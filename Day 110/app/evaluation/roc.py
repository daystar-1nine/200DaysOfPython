"""
ROC and Precision-Recall curve evaluation and plotting.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_roc_pr_curves(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    save_path: Path = None
) -> plt.Figure:
    """
    Renders side-by-side ROC and Precision-Recall curves for multiple models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, lw=2, label=f"{name} (AUC={pr_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title("ROC Curves", fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    baseline = float(np.mean(y_true))
    ax2.axhline(baseline, color="k", linestyle="--", alpha=0.5, label=f"Random ({baseline:.2f})")
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title("Precision-Recall Curves", fontweight="bold", fontsize=12)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig

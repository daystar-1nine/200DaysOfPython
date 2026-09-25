"""
Model evaluation visualization for Day 106: RNNs & Sequential Text Learning.
Generates Charts 9 to 14.
"""

from pathlib import Path
from typing import Dict, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_evaluation_charts(
    cm_dict: Dict[str, Union[int, float]],
    roc_dict: Dict[str, Union[float, np.ndarray]],
    pr_dict: Dict[str, Union[float, np.ndarray]],
    threshold_df: pd.DataFrame,
    output_dir: Union[str, Path]
) -> None:
    """Generate Charts 9 to 14 and save as PNG images."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 9. Confusion Matrix Heatmap
    matrix = np.array(cm_dict["matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Ham", "Pred Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Chart 9: Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "9_confusion_matrix.png", dpi=150)
    plt.close()

    # 10. ROC Curve
    plt.figure(figsize=(7, 5))
    plt.plot(roc_dict["fpr"], roc_dict["tpr"], color="#2980b9", lw=2.5,
             label=f"SimpleRNN (AUC = {roc_dict['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess")
    plt.title("Chart 10: Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "10_roc_curve.png", dpi=150)
    plt.close()

    # 11. Precision-Recall Curve
    plt.figure(figsize=(7, 5))
    plt.plot(pr_dict["recall"], pr_dict["precision"], color="#27ae60", lw=2.5,
             label=f"SimpleRNN (AP = {pr_dict['average_precision']:.4f})")
    plt.title("Chart 11: Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out / "11_precision_recall_curve.png", dpi=150)
    plt.close()

    # 12. Threshold vs F1
    plt.figure(figsize=(7, 5))
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", color="#8e44ad", lw=2.2)
    plt.title("Chart 12: Decision Threshold vs F1-Score", fontsize=14, fontweight="bold")
    plt.xlabel("Probability Decision Threshold")
    plt.ylabel("F1 Score")
    plt.ylim(0.0, 1.05)
    plt.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default (0.50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "12_threshold_vs_f1.png", dpi=150)
    plt.close()

    # 13. Threshold vs Precision
    plt.figure(figsize=(7, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="s", color="#d35400", lw=2.2)
    plt.title("Chart 13: Decision Threshold vs Precision", fontsize=14, fontweight="bold")
    plt.xlabel("Probability Decision Threshold")
    plt.ylabel("Precision (Avoid False Positives)")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(out / "13_threshold_vs_precision.png", dpi=150)
    plt.close()

    # 14. Threshold vs Recall
    plt.figure(figsize=(7, 5))
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="^", color="#16a085", lw=2.2)
    plt.title("Chart 14: Decision Threshold vs Recall", fontsize=14, fontweight="bold")
    plt.xlabel("Probability Decision Threshold")
    plt.ylabel("Recall (Avoid False Negatives)")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(out / "14_threshold_vs_recall.png", dpi=150)
    plt.close()

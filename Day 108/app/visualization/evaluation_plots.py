import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix

def generate_dataset_plots(df: pd.DataFrame, charts_dir: Path):
    """
    Generates Plots 1-4 for dataset exploration:
      Plot 1: Class distribution
      Plot 2: Message length distribution (characters)
      Plot 3: Token distribution (word count)
      Plot 4: Character count distribution grouped by class
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    counts = df["label"].value_counts()
    labels = ["Ham (0)", "Spam (1)"]
    vals = [counts.get(0, 0), counts.get(1, 0)]
    bars = plt.bar(labels, vals, color=["#2ca02c", "#d62728"], edgecolor="black", alpha=0.85)
    plt.title("Plot 1: SMS Class Distribution", fontsize=12, fontweight="bold")
    plt.ylabel("Count")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + max(vals)*0.01, f"{h}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(charts_dir / "01_class_distribution.png", dpi=200)
    plt.close()
    
    # Text lengths
    char_lens = df["text"].astype(str).str.len()
    token_lens = df["text"].astype(str).apply(lambda x: len(x.split()))
    
    # 2. Message length distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(char_lens, bins=40, kde=True, color="#1f77b4")
    plt.title("Plot 2: Message Character Length Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Characters per Message")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(charts_dir / "02_message_length_distribution.png", dpi=200)
    plt.close()
    
    # 3. Token distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(token_lens, bins=35, kde=True, color="#9467bd")
    plt.title("Plot 3: Message Token Count Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Tokens per Message")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(charts_dir / "03_token_distribution.png", dpi=200)
    plt.close()
    
    # 4. Character distribution by class
    plt.figure(figsize=(7, 4))
    df_temp = df.copy()
    df_temp["char_len"] = char_lens
    df_temp["Class"] = df_temp["label"].map({0: "Ham", 1: "Spam"})
    sns.boxplot(x="Class", y="char_len", data=df_temp, palette=["#2ca02c", "#d62728"])
    plt.title("Plot 4: Character Count by Class", fontsize=12, fontweight="bold")
    plt.ylabel("Character Length")
    plt.tight_layout()
    plt.savefig(charts_dir / "04_character_distribution_by_class.png", dpi=200)
    plt.close()

def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path):
    """
    Plots and saves annotated 2x2 confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Pred Ham", "Pred Spam"],
        yticklabels=["Actual Ham", "Actual Spam"]
    )
    plt.title(f"{model_name} Confusion Matrix", fontsize=11, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def generate_evaluation_plots(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    roc_data: Dict[str, Tuple],
    pr_data: Dict[str, Tuple],
    charts_dir: Path
):
    """
    Generates:
      Plot 11: RNN confusion matrix
      Plot 12: LSTM confusion matrix
      Plot 13: GRU confusion matrix
      Plot 14: ROC comparison
      Plot 15: Precision-Recall comparison
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrices
    for idx, name in enumerate(["RNN", "LSTM", "GRU"], start=11):
        if name in model_probs:
            y_pred = (model_probs[name] >= 0.5).astype(int)
            plot_confusion_matrix_heatmap(
                y_true, y_pred, name,
                charts_dir / f"{idx}_{name.lower()}_confusion_matrix.png"
            )
            
    # Plot 14: ROC Comparison
    plt.figure(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        c = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.title("Plot 14: ROC Curve Comparison across Recurrent Models", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(charts_dir / "14_roc_comparison.png", dpi=200)
    plt.close()
    
    # Plot 15: Precision-Recall Comparison
    plt.figure(figsize=(7, 5))
    for i, (name, (prec, rec, ap)) in enumerate(pr_data.items()):
        c = colors[i % len(colors)]
        plt.plot(rec, prec, color=c, lw=2, label=f"{name} (AP = {ap:.3f})")
    plt.title("Plot 15: Precision-Recall Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(charts_dir / "15_precision_recall_comparison.png", dpi=200)
    plt.close()

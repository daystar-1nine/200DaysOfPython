import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix

def generate_benchmark_visualizations(
    df_raw: pd.DataFrame,
    comparison_df: pd.DataFrame,
    histories: Dict[str, Dict],
    attn_weights: np.ndarray,
    token_attn_df: pd.DataFrame,
    entropies: np.ndarray,
    y_test: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    roc_data: Dict[str, Tuple],
    pr_data: Dict[str, Tuple],
    thresh_df: pd.DataFrame,
    tokenized_texts: list,
    charts_dir: Path
):
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # Plots 1-3: Dataset Exploratory
    # -------------------------------------------------------------
    # Plot 1: Class distribution
    plt.figure(figsize=(6, 4))
    counts = df_raw["label"].value_counts()
    bars = plt.bar(["Ham (0)", "Spam (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#2ca02c", "#d62728"], edgecolor="black", alpha=0.85)
    plt.title("Plot 1: SMS Class Distribution", fontsize=11, fontweight="bold")
    plt.ylabel("Message Count")
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 1, str(int(b.get_height())), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(charts_dir / "01_class_distribution.png", dpi=200)
    plt.close()
    
    # Plot 2: Message character length distribution
    char_lens = df_raw["text"].astype(str).str.len()
    plt.figure(figsize=(6.5, 4))
    sns.histplot(char_lens, bins=35, kde=True, color="#1f77b4")
    plt.title("Plot 2: Message Character Length Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Characters per Message")
    plt.tight_layout()
    plt.savefig(charts_dir / "02_message_length_distribution.png", dpi=200)
    plt.close()
    
    # Plot 3: Token length distribution
    tok_lens = df_raw["text"].astype(str).apply(lambda s: len(s.split()))
    plt.figure(figsize=(6.5, 4))
    sns.histplot(tok_lens, bins=30, kde=True, color="#9467bd")
    plt.title("Plot 3: Message Token Count Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Tokens per Message")
    plt.tight_layout()
    plt.savefig(charts_dir / "03_token_distribution.png", dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plots 8-13: Model Comparison Metrics
    # -------------------------------------------------------------
    models = comparison_df["model"].tolist()
    
    # Plot 8: Parameter Count
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, comparison_df["parameters"], color=sns.color_palette("muted", len(models)), edgecolor="black")
    plt.title("Plot 8: Parameter Count Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("Parameters")
    plt.xticks(rotation=20, ha="right")
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 500, f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / "08_parameter_count.png", dpi=200)
    plt.close()
    
    # Plot 9: Training Time
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, comparison_df["training_time_sec"], color="#ff7f0e", edgecolor="black", alpha=0.85)
    plt.title("Plot 9: Training Time Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("Seconds")
    plt.xticks(rotation=20, ha="right")
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 0.05, f"{b.get_height():.2f}s", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / "09_training_time.png", dpi=200)
    plt.close()
    
    # Plot 10: F1 Comparison
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, comparison_df["f1"], color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.title("Plot 10: F1 Score Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=20, ha="right")
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 0.02, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / "10_f1_comparison.png", dpi=200)
    plt.close()
    
    # Plot 11: Precision Comparison
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, comparison_df["precision"], color="#3b528b", edgecolor="black", alpha=0.85)
    plt.title("Plot 11: Precision Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("Precision")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "11_precision_comparison.png", dpi=200)
    plt.close()
    
    # Plot 12: Recall Comparison
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, comparison_df["recall"], color="#5ec962", edgecolor="black", alpha=0.85)
    plt.title("Plot 12: Recall Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("Recall")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "12_recall_comparison.png", dpi=200)
    plt.close()
    
    # Plot 13: ROC-AUC Comparison
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, comparison_df["roc_auc"], color="#fde725", edgecolor="black", alpha=0.85)
    plt.title("Plot 13: ROC-AUC Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("ROC-AUC")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "13_roc_auc_comparison.png", dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plots 14-16: Selected Attention Heatmaps (Spam, Ham, Misclass)
    # -------------------------------------------------------------
    if len(tokenized_texts) > 0 and attn_weights is not None:
        y_pred = (model_probs["GRU + Attention"] >= 0.5).astype(int)
        
        # Plot 14: Attention heatmap - spam
        spam_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        if len(spam_indices) > 0:
            idx = spam_indices[0]
            toks = tokenized_texts[idx][:len(attn_weights[idx])]
            wts = attn_weights[idx][:len(toks)]
            wts = wts / max(wts.sum(), 1e-9)
            plt.figure(figsize=(max(6, len(toks)*0.8), 2.5))
            sns.heatmap(np.expand_dims(wts, 0), annot=True, fmt=".2f", cmap="Reds", xticklabels=toks, yticklabels=["Attention"], cbar=False)
            plt.title("Plot 14: Attention Heatmap — Spam Example", fontsize=10, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(charts_dir / "14_attention_heatmap_spam.png", dpi=200)
            plt.close()
            
        # Plot 15: Attention heatmap - ham
        ham_indices = np.where((y_test == 0) & (y_pred == 0))[0]
        if len(ham_indices) > 0:
            idx = ham_indices[0]
            toks = tokenized_texts[idx][:len(attn_weights[idx])]
            wts = attn_weights[idx][:len(toks)]
            wts = wts / max(wts.sum(), 1e-9)
            plt.figure(figsize=(max(6, len(toks)*0.8), 2.5))
            sns.heatmap(np.expand_dims(wts, 0), annot=True, fmt=".2f", cmap="Blues", xticklabels=toks, yticklabels=["Attention"], cbar=False)
            plt.title("Plot 15: Attention Heatmap — Ham Example", fontsize=10, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(charts_dir / "15_attention_heatmap_ham.png", dpi=200)
            plt.close()
            
        # Plot 16: Attention heatmap - misclassification
        err_indices = np.where(y_test != y_pred)[0]
        idx = err_indices[0] if len(err_indices) > 0 else 0
        toks = tokenized_texts[idx][:len(attn_weights[idx])]
        wts = attn_weights[idx][:len(toks)]
        wts = wts / max(wts.sum(), 1e-9)
        plt.figure(figsize=(max(6, len(toks)*0.8), 2.5))
        sns.heatmap(np.expand_dims(wts, 0), annot=True, fmt=".2f", cmap="Purples", xticklabels=toks, yticklabels=["Attention"], cbar=False)
        plt.title(f"Plot 16: Attention Heatmap — Message Sample (Actual={y_test[idx]}, Pred={y_pred[idx]})", fontsize=10, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(charts_dir / "16_attention_heatmap_misclassification.png", dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # Plot 17: Attention Entropy Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(6.5, 4))
    sns.histplot(entropies, bins=15, kde=True, color="#e377c2")
    plt.title("Plot 17: Attention Entropy Distribution H(A)", fontsize=11, fontweight="bold")
    plt.xlabel("Entropy (Higher = Diffuse, Lower = Concentrated)")
    plt.tight_layout()
    plt.savefig(charts_dir / "17_attention_entropy_distribution.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Plot 18: Top-attention token frequency
    # -------------------------------------------------------------
    if not token_attn_df.empty:
        top10 = token_attn_df.head(10)
        plt.figure(figsize=(7, 4.5))
        bars = plt.barh(top10["token"][::-1], top10["average_attention"][::-1], color="#17becf")
        plt.title("Plot 18: Top-10 Attended Tokens by Average Weight", fontsize=11, fontweight="bold")
        plt.xlabel("Average Attention Weight")
        plt.tight_layout()
        plt.savefig(charts_dir / "18_top_attention_tokens.png", dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # Plots 19-20: Confusion Matrices (GRU & Attention)
    # -------------------------------------------------------------
    for idx_num, m_name in [(19, "GRU"), (20, "GRU + Attention")]:
        if m_name in model_probs:
            y_pred = (model_probs[m_name] >= 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Pred Ham", "Pred Spam"], yticklabels=["Actual Ham", "Actual Spam"])
            plt.title(f"Plot {idx_num}: Confusion Matrix — {m_name}", fontsize=11, fontweight="bold")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(charts_dir / f"{idx_num}_confusion_matrix_{m_name.lower().replace(' ', '_').replace('+', '')}.png", dpi=200)
            plt.close()

    # -------------------------------------------------------------
    # Plot 21: ROC Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.title("Plot 21: ROC Curve Comparison", fontsize=11, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(charts_dir / "21_roc_curves.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Plot 22: Precision-Recall Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    for i, (name, (prec, rec, ap)) in enumerate(pr_data.items()):
        plt.plot(rec, prec, color=colors[i % len(colors)], lw=2, label=f"{name} (AP = {ap:.3f})")
    plt.title("Plot 22: Precision-Recall Curve Comparison", fontsize=11, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(charts_dir / "22_precision_recall_curves.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Plot 23: Threshold vs F1
    # -------------------------------------------------------------
    if not thresh_df.empty:
        plt.figure(figsize=(7, 4.5))
        for m_name, grp in thresh_df.groupby("model"):
            plt.plot(grp["threshold"], grp["f1"], marker="o", lw=2, label=m_name)
        plt.title("Plot 23: Decision Threshold vs F1 Score", fontsize=11, fontweight="bold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / "23_threshold_vs_f1.png", dpi=200)
        plt.close()

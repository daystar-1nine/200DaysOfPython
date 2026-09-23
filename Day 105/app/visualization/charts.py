"""
Charts and visual analytics module for Day 105: Neural NLP & Text Classification.
Generates all 16 required publication-quality plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class Visualizer:
    """Generates all 16 required visualizations for neural text classification."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    def plot_class_distribution(self, df: pd.DataFrame, filename: str = "1_class_distribution.png") -> Path:
        """Chart 1: Class distribution (ham vs spam)."""
        counts = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, palette=["#2ecc71", "#e74c3c"], legend=False)
        ax.set_title("1. SMS Dataset Class Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            pct = (v / len(df)) * 100
            ax.text(i, v + 5, f"{v} ({pct:.1f}%)", ha="center", fontweight="bold")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_message_length_distribution(self, df: pd.DataFrame, filename: str = "2_message_length_distribution.png") -> Path:
        """Chart 2: Raw message character length distribution."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=df["text"].str.len(), hue="label", kde=True, ax=ax, palette=["#2ecc71", "#e74c3c"], bins=30)
        ax.set_title("2. Message Character Length Distribution by Class", fontsize=12, fontweight="bold")
        ax.set_xlabel("Character Length")
        ax.set_ylabel("Frequency")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_sequence_length_distribution(self, token_lengths: List[int], max_len: int = 40, filename: str = "3_sequence_length_distribution.png") -> Path:
        """Chart 3: Sequence token length distribution with truncation threshold."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(token_lengths, kde=True, ax=ax, color="#3498db", bins=25)
        ax.axvline(max_len, color="#e74c3c", linestyle="--", linewidth=2, label=f"Max Length Cutoff ({max_len})")
        truncated_pct = (sum(1 for l in token_lengths if l > max_len) / len(token_lengths)) * 100
        ax.set_title(f"3. Sequence Token Lengths ({truncated_pct:.1f}% Truncated at {max_len})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Frequency")
        ax.legend()

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_vocabulary_frequency(self, word_counts: Dict[str, int], top_n: int = 25, filename: str = "4_vocabulary_frequency.png") -> Path:
        """Chart 4: Vocabulary frequency distribution (Top N words)."""
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, counts = zip(*top_words)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words), hue=list(words), ax=ax, palette="mako", legend=False)
        ax.set_title(f"4. Top {top_n} Most Frequent Vocabulary Words", fontsize=12, fontweight="bold")
        ax.set_xlabel("Occurrences")
        ax.set_ylabel("Token")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_training_loss(self, history: Dict[str, List[float]], filename: str = "5_training_loss.png") -> Path:
        """Chart 5: Training loss curve across epochs."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["epoch"], history["train_loss"], marker="o", color="#2980b9", label="Training Loss")
        ax.set_title("5. Training Loss Curve (BCE Loss)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_validation_loss(self, history: Dict[str, List[float]], filename: str = "6_validation_loss.png") -> Path:
        """Chart 6: Validation loss curve across epochs."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["epoch"], history["val_loss"], marker="s", color="#e67e22", label="Validation Loss")
        min_loss_idx = int(np.argmin(history["val_loss"]))
        min_loss = history["val_loss"][min_loss_idx]
        ax.scatter([history["epoch"][min_loss_idx]], [min_loss], color="red", s=100, zorder=5, label=f"Best: {min_loss:.4f}")
        ax.set_title("6. Validation Loss Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend()

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_training_accuracy(self, history: Dict[str, List[float]], filename: str = "7_training_accuracy.png") -> Path:
        """Chart 7: Training accuracy curve across epochs."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["epoch"], history["train_acc"], marker="o", color="#27ae60", label="Training Accuracy")
        ax.set_title("7. Training Accuracy Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.5, 1.02)
        ax.legend()

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_validation_accuracy(self, history: Dict[str, List[float]], filename: str = "8_validation_accuracy.png") -> Path:
        """Chart 8: Validation accuracy curve across epochs."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["epoch"], history["val_acc"], marker="s", color="#8e44ad", label="Validation Accuracy")
        max_acc_idx = int(np.argmax(history["val_acc"]))
        max_acc = history["val_acc"][max_acc_idx]
        ax.scatter([history["epoch"][max_acc_idx]], [max_acc], color="green", s=100, zorder=5, label=f"Peak: {max_acc:.4f}")
        ax.set_title("8. Validation Accuracy Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.5, 1.02)
        ax.legend()

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_confusion_matrix(self, cm_dict: Dict[str, int], filename: str = "9_confusion_matrix.png") -> Path:
        """Chart 9: Confusion matrix heatmap."""
        matrix = np.array([
            [cm_dict["true_negatives"], cm_dict["false_positives"]],
            [cm_dict["false_negatives"], cm_dict["true_positives"]]
        ])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Ham", "Pred Spam"], yticklabels=["Actual Ham", "Actual Spam"])
        ax.set_title("9. Test Set Confusion Matrix", fontsize=12, fontweight="bold")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, filename: str = "10_roc_curve.png") -> Path:
        """Chart 10: Receiver Operating Characteristic (ROC) curve."""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_score = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"Neural Classifier (AUC = {roc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_title("10. Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_probs: np.ndarray, filename: str = "11_precision_recall_curve.png") -> Path:
        """Chart 11: Precision-Recall curve."""
        prec, rec, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, color="#16a085", lw=2, label=f"Neural Classifier (AP = {ap:.4f})")
        ax.set_title("11. Precision-Recall Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_model_comparison(self, comparison_df: pd.DataFrame, filename: str = "12_model_comparison.png") -> Path:
        """Chart 12: Model comparison across F1 and ROC-AUC metrics."""
        fig, ax = plt.subplots(figsize=(9, 5))
        df_melt = pd.melt(comparison_df, id_vars=["model"], value_vars=["f1", "roc_auc"],
                          var_name="metric", value_name="score")
        sns.barplot(data=df_melt, x="model", y="score", hue="metric", ax=ax, palette="Set2")
        ax.set_title("12. Model Comparison: Classical vs Neural Architectures", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.1)

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                            textcoords="offset points")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_embedding_pca(self, pca_df: pd.DataFrame, filename: str = "13_embedding_pca.png") -> Path:
        """Chart 13: 2D PCA projection of learned word embeddings."""
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(pca_df["pc1"], pca_df["pc2"], color="#8e44ad", alpha=0.7, s=60)
        for _, row in pca_df.iterrows():
            ax.annotate(row["word"], (row["pc1"], row["pc2"]),
                        fontsize=9, alpha=0.85, xytext=(3, 3), textcoords="offset points")
        ax.set_title("13. 2D PCA Projection of Learned Word Embeddings", fontsize=12, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_error_distribution(self, error_df: pd.DataFrame, filename: str = "14_error_distribution.png") -> Path:
        """Chart 14: Error distribution breakdown (False Positives vs False Negatives)."""
        fig, ax = plt.subplots(figsize=(6, 4))
        if error_df.empty:
            ax.text(0.5, 0.5, "Zero Misclassifications\n(100% Test Accuracy)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#27ae60")
            ax.set_title("14. Classification Error Breakdown", fontsize=12, fontweight="bold")
            ax.axis("off")
        else:
            counts = error_df["error_type"].value_counts()
            sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, palette=["#e74c3c", "#f39c12"], legend=False)
            ax.set_title("14. Classification Error Breakdown", fontsize=12, fontweight="bold")
            ax.set_xlabel("Error Type")
            ax.set_ylabel("Count")
            for i, v in enumerate(counts.values):
                ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_false_positive_examples(self, error_df: pd.DataFrame, filename: str = "15_false_positive_examples.png") -> Path:
        """Chart 15: False positive examples table."""
        fps = error_df[error_df["error_type"] == "False Positive"].head(5)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.set_title("15. False Positive Samples (Ham classified as Spam)", fontsize=12, fontweight="bold", pad=20)

        table_data = []
        for _, row in fps.iterrows():
            snip = row["text"][:75] + ("..." if len(row["text"]) > 75 else "")
            table_data.append([snip, f"{row['probability']:.3f}"])

        if not table_data:
            table_data = [["No False Positives observed on test set.", "0.000"]]

        table = ax.table(cellText=table_data, colLabels=["Message Text", "Spam Prob"],
                         loc="center", cellLoc="left", colWidths=[0.8, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_false_negative_examples(self, error_df: pd.DataFrame, filename: str = "16_false_negative_examples.png") -> Path:
        """Chart 16: False negative examples table."""
        fns = error_df[error_df["error_type"] == "False Negative"].head(5)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.set_title("16. False Negative Samples (Spam classified as Ham)", fontsize=12, fontweight="bold", pad=20)

        table_data = []
        for _, row in fns.iterrows():
            snip = row["text"][:75] + ("..." if len(row["text"]) > 75 else "")
            table_data.append([snip, f"{row['probability']:.3f}"])

        if not table_data:
            table_data = [["No False Negatives observed on test set.", "0.000"]]

        table = ax.table(cellText=table_data, colLabels=["Message Text", "Spam Prob"],
                         loc="center", cellLoc="left", colWidths=[0.8, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        out = self.output_dir / filename
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

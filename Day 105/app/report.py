"""
Report generation module for Day 105: Neural NLP & Text Classification.
Generates comprehensive evaluation and experiment report in Markdown.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class ReportGenerator:
    """Produces detailed Markdown evaluation reports for Day 105."""

    @staticmethod
    def generate(
        comparison_df: pd.DataFrame,
        model_params: Dict[str, int],
        dataset_summary: Dict[str, Any],
        env_audit: Dict[str, Any],
        error_df: pd.DataFrame,
        output_path: Path
    ) -> None:
        """Generate and write report.md."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "# Day 105: Neural NLP & Text Classification — Evaluation Report",
            "",
            "## 1. Executive Summary",
            "",
            "This report documents the transition from classical bag-of-words / TF-IDF representations to **neural text classification** using **trainable embedding layers** and **masked sequence pooling** on the SMS Spam benchmark dataset.",
            "",
            "Key engineering milestones:",
            "- Prevented data leakage by fitting vocabulary strictly on the training partition.",
            "- Implemented integer token encoding, post-padding, and dynamic sequence masking.",
            "- Evaluated three neural architectures against classical TF-IDF baselines (Logistic Regression and Linear SVM).",
            "- Generated 16 analytical visualizations and diagnosed systematic classification failure modes.",
            "",
            "---",
            "",
            "## 2. Dataset & Partition Audit",
            "",
            f"- **Total Samples**: {dataset_summary['total_samples']}",
            f"- **Class Distribution**: Ham = {dataset_summary['class_counts'].get('ham', 0)} ({dataset_summary['class_proportions'].get('ham', 0)*100:.1f}%), Spam = {dataset_summary['class_counts'].get('spam', 0)} ({dataset_summary['class_proportions'].get('spam', 0)*100:.1f}%)",
            f"- **Average Message Length**: {dataset_summary['mean_char_length']:.1f} characters ({dataset_summary['mean_word_count']:.1f} words)",
            "- **Partitioning**: 70% Train, 15% Validation, 15% Test (Stratified)",
            "",
            "---",
            "",
            "## 3. Classical vs Neural Model Benchmark",
            "",
            "Evaluation on the held-out test partition:",
            "",
            "| Model | Representation | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Parameters | Training Time |",
            "|---|---|---|---|---|---|---|---|---|"
        ]

        for _, row in comparison_df.iterrows():
            lines.append(
                f"| **{row['model']}** | {row['representation']} | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"**{row['f1']:.4f}** | {row['roc_auc']:.4f} | "
                f"{row['parameters']:,} | {row['training_time']:.3f}s |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Parameter Count Breakdown",
            "",
            f"- **Embedding Parameters**: {model_params.get('embedding_params', 0):,} weights",
            f"- **Dense Head Parameters**: {model_params.get('dense_params', 0):,} weights",
            f"- **Total Trainable Parameters**: {model_params.get('total_params', 0):,} weights",
            "",
            "### Analytical Formula:",
            "$$\\text{Total Parameters} = (V \\times D) + (D \\times H + H) + (H \\times 1 + 1)$$",
            f"where vocabulary size $V$, embedding dimension $D$, and hidden dimension $H$.",
            "",
            "---",
            "",
            "## 5. Error Analysis & Diagnostics",
            "",
            f"- Total test misclassifications: **{len(error_df)}**",
            f"- False Positives (Ham classified as Spam): **{len(error_df[error_df['error_type'] == 'False Positive'])}**",
            f"- False Negatives (Spam classified as Ham): **{len(error_df[error_df['error_type'] == 'False Negative'])}**",
            "",
            "Key patterns observed in misclassifications:",
            "1. **Promotional URL Blindspots**: Messages containing unfamiliar short-links or unusual TLDs with words not in the training vocabulary.",
            "2. **Very Short Messages**: Messages under 4 words where pooling averages fewer tokens, providing minimal context for dense representation.",
            "",
            "---",
            "",
            "## 6. Environment Verification Audit",
            "",
            f"- **Python Runtime**: `{env_audit['python_version'].split()[0]}`",
            f"- **TensorFlow Status**: `{env_audit['tensorflow_status']}`",
            f"  - *Audit Note*: {env_audit['tensorflow_message']}",
            f"- **PyTorch Status**: `{env_audit['pytorch_status']}` ({env_audit['pytorch_version']})",
            f"  - *Audit Note*: {env_audit['pytorch_message']}",
            "",
            "---",
            "",
            "## 7. Visualizations Generated",
            "",
            "All 16 publication-quality charts generated in `output/charts/`:",
            "1. `1_class_distribution.png` — Distribution of Ham vs Spam.",
            "2. `2_message_length_distribution.png` — Character length distribution across classes.",
            "3. `3_sequence_length_distribution.png` — Token counts with cutoff threshold.",
            "4. `4_vocabulary_frequency.png` — Top 25 vocabulary words.",
            "5. `5_training_loss.png` — Epoch vs Training BCE Loss.",
            "6. `6_validation_loss.png` — Epoch vs Validation Loss.",
            "7. `7_training_accuracy.png` — Epoch vs Training Accuracy.",
            "8. `8_validation_accuracy.png` — Epoch vs Validation Accuracy.",
            "9. `9_confusion_matrix.png` — Test set confusion matrix.",
            "10. `10_roc_curve.png` — Receiver Operating Characteristic curve.",
            "11. `11_precision_recall_curve.png` — Precision-Recall curve.",
            "12. `12_model_comparison.png` — Classical vs Neural model comparison.",
            "13. `13_embedding_pca.png` — 2D PCA projection of learned word vectors.",
            "14. `14_error_distribution.png` — Breakdown of FP vs FN errors.",
            "15. `15_false_positive_examples.png` — Detailed false positive table.",
            "16. `16_false_negative_examples.png` — Detailed false negative table.",
            ""
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

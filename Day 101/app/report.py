"""Markdown report generator."""
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def generate_markdown_report(
    metrics: Dict[str, float],
    cm_breakdown: Dict[str, int],
    model_comp_df: pd.DataFrame,
    diagnostics: List[Dict[str, Any]],
    output_path: Path
):
    # Format table manually to avoid tabulate dependency
    headers = list(model_comp_df.columns)
    table_md = "| " + " | ".join(headers) + " |\n"
    table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in model_comp_df.iterrows():
        table_md += "| " + " | ".join([f"{val:.4f}" if isinstance(val, (float, int)) and not isinstance(val, bool) and not isinstance(val, str) else str(val) for val in row]) + " |\n"

    md = f"""# Day 101 — NLP Text Classification & Text Processing Report

## 1. Executive Summary
- **Primary Task**: Binary Text Classification (SMS Spam Detection: `ham` vs `spam`).
- **Primary Model**: Logistic Regression with TF-IDF Vectorization.
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1']:.4f}
- **ROC-AUC**: {metrics.get('roc_auc', 0.0):.4f}
- **Average Precision (PR-AUC)**: {metrics.get('avg_precision', 0.0):.4f}

---

## 2. Confusion Matrix Breakdown
- **True Positives (Spam correctly caught)**: {cm_breakdown['true_positives']}
- **True Negatives (Ham correctly preserved)**: {cm_breakdown['true_negatives']}
- **False Positives (Ham incorrectly blocked)**: {cm_breakdown['false_positives']}
- **False Negatives (Spam missed)**: {cm_breakdown['false_negatives']}

> **Spam Detection Trade-off**: In spam filtering, **False Positives** are significantly more damaging than False Negatives, because legitimate important messages (e.g. flight tickets, family updates) should not be sent to spam.

---

## 3. Model Benchmark Comparison
{table_md}

---

## 4. Error Analysis & Diagnostics
Misclassified samples analyzed: {len(diagnostics)}

"""
    for idx, diag in enumerate(diagnostics[:8], 1):
        md += f"### Example {idx} [{diag['error_type']}]\n"
        md += f"- **Message**: `{diag['text']}`\n"
        md += f"- **Spam Probability**: {diag['probability']:.4f}\n"
        reasons_str = ", ".join(diag['reasons'])
        md += f"- **Identified Reasons**: {reasons_str}\n\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated report at {output_path}")

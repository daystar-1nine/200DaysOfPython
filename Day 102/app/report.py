"""Markdown report generator for Day 102 Benchmark."""
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def generate_benchmark_report(
    audit_data: Dict[str, Any],
    benchmark_df: pd.DataFrame,
    feature_exp_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_path: Path
):
    headers_b = list(benchmark_df.columns)
    table_b = "| " + " | ".join(headers_b) + " |\n"
    table_b += "| " + " | ".join(["---"] * len(headers_b)) + " |\n"
    for _, row in benchmark_df.iterrows():
        table_b += "| " + " | ".join([f"{val:.4f}" if isinstance(val, (float, int)) and not isinstance(val, bool) and not isinstance(val, str) else str(val) for val in row]) + " |\n"

    headers_f = list(feature_exp_df.columns)
    table_f = "| " + " | ".join(headers_f) + " |\n"
    table_f += "| " + " | ".join(["---"] * len(headers_f)) + " |\n"
    for _, row in feature_exp_df.iterrows():
        table_f += "| " + " | ".join([f"{val:.4f}" if isinstance(val, (float, int)) and not isinstance(val, bool) and not isinstance(val, str) else str(val) for val in row]) + " |\n"

    md = f"""# Day 102 — NLP Classification Benchmark & Feature Engineering Report

## 1. Dataset Audit
- **Total Records**: {audit_data['total_records']}
- **Class Distribution**: {audit_data['class_distribution']}
- **Duplicate Count**: {audit_data['duplicate_count']}
- **Missing Values**: {audit_data['missing_values']}
- **Message Length (Chars)**: Mean={audit_data['char_length']['mean']:.1f}, Median={audit_data['char_length']['median']:.1f}, Min={audit_data['char_length']['min']}, Max={audit_data['char_length']['max']}
- **Word Count**: Mean={audit_data['word_count']['mean']:.1f}, Median={audit_data['word_count']['median']:.1f}, Min={audit_data['word_count']['min']}, Max={audit_data['word_count']['max']}

---

## 2. Model Benchmark Table (5-Fold Stratified CV + Test Metrics)
{table_b}

---

## 3. Feature Engineering Experiments
{table_f}

---

## 4. Error Analysis
- **Total Errors Recorded Across Test Set**: {len(errors_df)}
"""
    if not errors_df.empty:
        headers_e = list(errors_df.columns)
        table_e = "| " + " | ".join(headers_e) + " |\n"
        table_e += "| " + " | ".join(["---"] * len(headers_e)) + " |\n"
        for _, row in errors_df.head(10).iterrows():
            table_e += "| " + " | ".join([str(val) for val in row]) + " |\n"
        md += "\n" + table_e
    else:
        md += "\n> No classification errors were observed on the held-out test split for the top linear pipelines!\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated benchmark report at {output_path}")

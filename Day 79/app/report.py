import os
from typing import Dict, Any

def generate_markdown_report(
    benchmark_metrics: Dict[str, Dict[str, float]],
    cv_summary: Dict[str, Dict[str, Dict[str, float]]],
    oob_score: float,
    best_params: Dict[str, Any],
    top_mdi_df,
    cost_summary: Dict[str, Any],
    output_file: str
):
    md = f"""# 🌲 Day 79: Customer Churn Random Forest Prediction Engine Report

## 1. Executive Summary
This report summarizes the results of Day 79 of the 200 Days of Python Challenge. We developed, tuned, and evaluated an ensemble **Random Forest Classifier** against a single **Decision Tree** and a **Baseline Logistic Regression** model for customer churn prediction.

---

## 2. Model Performance Benchmarks (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| **Logistic Baseline** | {benchmark_metrics['Logistic Regression']['accuracy']:.4f} | {benchmark_metrics['Logistic Regression']['precision']:.4f} | {benchmark_metrics['Logistic Regression']['recall']:.4f} | {benchmark_metrics['Logistic Regression']['f1_score']:.4f} | {benchmark_metrics['Logistic Regression']['roc_auc']:.4f} | {benchmark_metrics['Logistic Regression']['pr_auc']:.4f} |
| **Decision Tree** | {benchmark_metrics['Decision Tree']['accuracy']:.4f} | {benchmark_metrics['Decision Tree']['precision']:.4f} | {benchmark_metrics['Decision Tree']['recall']:.4f} | {benchmark_metrics['Decision Tree']['f1_score']:.4f} | {benchmark_metrics['Decision Tree']['roc_auc']:.4f} | {benchmark_metrics['Decision Tree']['pr_auc']:.4f} |
| **Random Forest (100 Trees)** | {benchmark_metrics['Random Forest']['accuracy']:.4f} | {benchmark_metrics['Random Forest']['precision']:.4f} | {benchmark_metrics['Random Forest']['recall']:.4f} | {benchmark_metrics['Random Forest']['f1_score']:.4f} | {benchmark_metrics['Random Forest']['roc_auc']:.4f} | {benchmark_metrics['Random Forest']['pr_auc']:.4f} |
| **Tuned Random Forest** | {benchmark_metrics['Tuned Random Forest']['accuracy']:.4f} | {benchmark_metrics['Tuned Random Forest']['precision']:.4f} | {benchmark_metrics['Tuned Random Forest']['recall']:.4f} | {benchmark_metrics['Tuned Random Forest']['f1_score']:.4f} | {benchmark_metrics['Tuned Random Forest']['roc_auc']:.4f} | {benchmark_metrics['Tuned Random Forest']['pr_auc']:.4f} |

- **Random Forest Out-of-Bag (OOB) Accuracy Score**: `{oob_score:.4f}`
- **Best Hyperparameters (GridSearchCV)**: `{best_params}`

---

## 3. 5-Fold Stratified Cross-Validation

| Model | Mean F1 Score | F1 Std Dev | Mean ROC AUC | ROC AUC Std Dev |
|---|---|---|---|---|
| **Logistic Baseline** | {cv_summary['Logistic Regression']['f1']['mean']:.4f} | ±{cv_summary['Logistic Regression']['f1']['std']:.4f} | {cv_summary['Logistic Regression']['roc_auc']['mean']:.4f} | ±{cv_summary['Logistic Regression']['roc_auc']['std']:.4f} |
| **Decision Tree** | {cv_summary['Decision Tree']['f1']['mean']:.4f} | ±{cv_summary['Decision Tree']['f1']['std']:.4f} | {cv_summary['Decision Tree']['roc_auc']['mean']:.4f} | ±{cv_summary['Decision Tree']['roc_auc']['std']:.4f} |
| **Random Forest** | {cv_summary['Random Forest']['f1']['mean']:.4f} | ±{cv_summary['Random Forest']['f1']['std']:.4f} | {cv_summary['Random Forest']['roc_auc']['mean']:.4f} | ±{cv_summary['Random Forest']['roc_auc']['std']:.4f} |

---

## 4. Top 10 Churn Drivers (MDI Feature Importance)

| Rank | Feature Name | Gini Importance | Tree-Level Std |
|---|---|---|---|
"""
    for idx, row in top_mdi_df.head(10).iterrows():
        md += f"| {idx+1} | `{row['Feature']}` | {row['Importance']:.4f} | ±{row['Std']:.4f} |\n"

    md += f"""
---

## 5. Business Cost Analysis & Financial Impact
- **False Positive Cost (Wasted Incentive)**: INR {cost_summary['cost_fp']:.2f}
- **False Negative Cost (Lost Customer Churn)**: INR {cost_summary['cost_fn']:.2f}
- **Unmanaged Churn Cost**: INR {cost_summary['unmanaged_cost']:,.2f}
- **Default Threshold (0.50) Cost**: INR {cost_summary['default_cost']:,.2f}
- **Optimal Decision Threshold**: `{cost_summary['optimal_threshold']:.2f}`
- **Optimal Threshold Cost**: INR {cost_summary['optimal_cost']:,.2f}
- **Net Cost Savings**: **INR {cost_summary['savings']:,.2f}** ({cost_summary['pct_savings']:.2f}% reduction)

---

## 6. Generated Analytical Artifacts
All 14 visual charts have been saved to `Day 79/output/charts/`:
1. `01_confusion_matrix_comparison.png`
2. `02_roc_curves_comparison.png`
3. `03_pr_curves_comparison.png`
4. `04_model_metrics_benchmark.png`
5. `05_feature_importance_mdi.png`
6. `06_permutation_importance.png`
7. `07_mdi_vs_permutation_comparison.png`
8. `08_threshold_vs_f1_precision_recall.png`
9. `09_business_cost_curve.png`
10. `10_oob_error_vs_trees.png`
11. `11_tree_variance_reduction.png`
12. `12_stratified_cv_boxplots.png`
13. `13_customer_risk_distribution.png`
14. `14_cost_savings_waterfall.png`
"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Report written to {output_file}")

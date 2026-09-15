import os
from typing import Dict, Any
import pandas as pd

def generate_milestone_markdown_report(
    df_comparison: pd.DataFrame,
    champion_model: str,
    top_features_df: pd.DataFrame,
    cost_summary: Dict[str, Any],
    output_path: str
):
    champ_row = df_comparison[df_comparison['Model'] == champion_model].iloc[0]
    
    md = f"""# 🏆 Day 80: End-to-End Machine Learning Model Selection Milestone Report

## 1. Executive Summary & Problem Formulation
This milestone report presents the complete end-to-end model selection engine for **Customer Churn Prediction**. We systematically formulated the business objective, prevented data leakage via isolated pipelines, benchmarked four diverse candidate algorithms, performed 5-fold stratified cross-validation and hyperparameter optimization, and calibrated the decision threshold against asymmetric business costs ($C_{{FP}} = \text{{INR }} 300, C_{{FN}} = \text{{INR }} 2,000$).

**Champion Model Selected for Production Deployment**: `{champion_model}`

---

## 2. Model Selection Benchmark Matrix

| Model Architecture | CV ROC-AUC | CV AP | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test AP | Test Business Cost |
|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in df_comparison.iterrows():
        md += f"| **{r['Model']}** | {r['CV_ROC_AUC_Mean']:.4f} (±{r['CV_ROC_AUC_Std']:.3f}) | {r['CV_AP_Mean']:.4f} | {r['Test_Accuracy']:.4f} | {r['Test_Precision']:.4f} | {r['Test_Recall']:.4f} | {r['Test_F1']:.4f} | {r['Test_ROC_AUC']:.4f} | {r['Test_AP']:.4f} | INR {r['Test_Business_Cost']:,.0f} |\n"

    md += f"""
---

## 3. Top Churn Predictors (Feature Attribution)

| Rank | Feature Name | MDI Importance | Standard Deviation |
|---|---|---|---|
"""
    for idx, r in top_features_df.head(10).iterrows():
        std_str = f"±{r['Std']:.4f}" if 'Std' in r else "N/A"
        md += f"| {idx+1} | `{r['Feature']}` | {r['Importance']:.4f} | {std_str} |\n"

    md += f"""
---

## 4. Business Cost & Threshold Optimization
- **Standard Threshold (0.50) Cost**: INR {cost_summary.get('default_cost', 0.0):,.2f}
- **Cost-Optimal Threshold**: `{cost_summary.get('optimal_threshold', 0.50):.2f}`
- **Optimal Cost**: INR {cost_summary.get('optimal_cost', 0.0):,.2f}
- **Net Cost Savings**: **INR {cost_summary.get('savings', 0.0):,.2f}** ({cost_summary.get('pct_savings', 0.0):.1f}% reduction)

---

## 5. Architectural Deliverables & Visual Artifacts
All 17 analytical figures are stored in `Day 80/output/charts/`:
1. `01_churn_distribution.png`
2. `02_churn_by_contract.png`
3. `03_churn_by_internet_service.png`
4. `04_tenure_distribution.png`
5. `05_monthly_charges_by_churn.png`
6. `06_cv_roc_auc_comparison.png`
7. `07_cv_average_precision_comparison.png`
8. `08_test_metric_comparison.png`
9. `09_train_vs_val_performance.png`
10. `10_confusion_matrix_best_model.png`
11. `11_roc_curves_all_models.png`
12. `12_precision_recall_curves.png`
13. `13_threshold_vs_precision_recall.png`
14. `14_threshold_vs_business_cost.png`
15. `15_feature_importance_mdi.png`
16. `16_permutation_importance.png`
17. `17_probability_calibration_curve.png`
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Milestone report written to {output_path}")

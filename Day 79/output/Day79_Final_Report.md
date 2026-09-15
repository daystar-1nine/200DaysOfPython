# 🌲 Day 79: Customer Churn Random Forest Prediction Engine Report

## 1. Executive Summary
This report summarizes the results of Day 79 of the 200 Days of Python Challenge. We developed, tuned, and evaluated an ensemble **Random Forest Classifier** against a single **Decision Tree** and a **Baseline Logistic Regression** model for customer churn prediction.

---

## 2. Model Performance Benchmarks (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| **Logistic Baseline** | 0.8360 | 0.6146 | 0.9365 | 0.7421 | 0.9388 | 0.8213 |
| **Decision Tree** | 0.9720 | 0.9242 | 0.9683 | 0.9457 | 0.9848 | 0.9426 |
| **Random Forest (100 Trees)** | 0.9680 | 0.9297 | 0.9444 | 0.9370 | 0.9825 | 0.9564 |
| **Tuned Random Forest** | 0.9680 | 0.9365 | 0.9365 | 0.9365 | 0.9790 | 0.9570 |

- **Random Forest Out-of-Bag (OOB) Accuracy Score**: `0.9590`
- **Best Hyperparameters (GridSearchCV)**: `{'classifier__max_depth': 12, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 50}`

---

## 3. 5-Fold Stratified Cross-Validation

| Model | Mean F1 Score | F1 Std Dev | Mean ROC AUC | ROC AUC Std Dev |
|---|---|---|---|---|
| **Logistic Baseline** | 0.7518 | ±0.0215 | 0.9423 | ±0.0043 |
| **Decision Tree** | 0.9137 | ±0.0193 | 0.9714 | ±0.0148 |
| **Random Forest** | 0.9162 | ±0.0247 | 0.9758 | ±0.0059 |

---

## 4. Top 10 Churn Drivers (MDI Feature Importance)

| Rank | Feature Name | Gini Importance | Tree-Level Std |
|---|---|---|---|
| 1 | `Contract_Type_One year` | 0.1961 | ±0.0462 |
| 2 | `Contract_Type_Two year` | 0.1625 | ±0.0496 |
| 3 | `Tenure_Months` | 0.1484 | ±0.1094 |
| 4 | `Support_Calls` | 0.0845 | ±0.0394 |
| 5 | `Tenure_to_Age_Ratio` | 0.0838 | ±0.0913 |
| 6 | `Total_Charges` | 0.0779 | ±0.0804 |
| 7 | `Support_Per_Month` | 0.0682 | ±0.0619 |
| 8 | `Complaints` | 0.0417 | ±0.0169 |
| 9 | `Monthly_Charges` | 0.0321 | ±0.0175 |
| 10 | `Usage_Hours` | 0.0221 | ±0.0107 |

---

## 5. Business Cost Analysis & Financial Impact
- **False Positive Cost (Wasted Incentive)**: INR 300.00
- **False Negative Cost (Lost Customer Churn)**: INR 2000.00
- **Unmanaged Churn Cost**: INR 252,000.00
- **Default Threshold (0.50) Cost**: INR 16,700.00
- **Optimal Decision Threshold**: `0.30`
- **Optimal Threshold Cost**: INR 12,000.00
- **Net Cost Savings**: **INR 4,700.00** (28.14% reduction)

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

# 🏆 Day 80: End-to-End Machine Learning Model Selection Milestone Report

## 1. Executive Summary & Problem Formulation
This milestone report presents the complete end-to-end model selection engine for **Customer Churn Prediction**. We systematically formulated the business objective, prevented data leakage via isolated pipelines, benchmarked four diverse candidate algorithms, performed 5-fold stratified cross-validation and hyperparameter optimization, and calibrated the decision threshold against asymmetric business costs ($C_{FP} = 	ext{INR } 300, C_{FN} = 	ext{INR } 2,000$).

**Champion Model Selected for Production Deployment**: `Decision Tree`

---

## 2. Model Selection Benchmark Matrix

| Model Architecture | CV ROC-AUC | CV AP | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test AP | Test Business Cost |
|---|---|---|---|---|---|---|---|---|---|
| **Decision Tree** | 0.9771 (±0.009) | 0.9331 | 0.9680 | 0.9044 | 0.9762 | 0.9389 | 0.9897 | 0.9516 | INR 9,900 |
| **Random Forest** | 0.9816 (±0.004) | 0.9302 | 0.9740 | 0.9313 | 0.9683 | 0.9494 | 0.9843 | 0.9487 | INR 10,700 |
| **Logistic Regression** | 0.9431 (±0.005) | 0.8504 | 0.8420 | 0.6243 | 0.9365 | 0.7492 | 0.9409 | 0.8322 | INR 37,300 |
| **Baseline (Dummy)** | 0.5000 (±0.000) | 0.2530 | 0.7480 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2520 | INR 252,000 |

---

## 3. Top Churn Predictors (Feature Attribution)

| Rank | Feature Name | MDI Importance | Standard Deviation |
|---|---|---|---|
| 1 | `Tenure_Months` | 0.2318 | ±0.1057 |
| 2 | `Contract_Type_One year` | 0.2187 | ±0.0566 |
| 3 | `Contract_Type_Two year` | 0.1850 | ±0.0551 |
| 4 | `Support_Calls` | 0.1043 | ±0.0421 |
| 5 | `Total_Charges` | 0.1032 | ±0.0950 |
| 6 | `Complaints` | 0.0478 | ±0.0223 |
| 7 | `Monthly_Charges` | 0.0331 | ±0.0176 |
| 8 | `Usage_Hours` | 0.0201 | ±0.0117 |
| 9 | `Age` | 0.0152 | ±0.0106 |
| 10 | `Late_Payments` | 0.0151 | ±0.0114 |

---

## 4. Business Cost & Threshold Optimization
- **Standard Threshold (0.50) Cost**: INR 9,900.00
- **Cost-Optimal Threshold**: `0.43`
- **Optimal Cost**: INR 9,900.00
- **Net Cost Savings**: **INR 0.00** (0.0% reduction)

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

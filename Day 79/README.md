# 🌲 Day 79: Random Forest & Ensemble Learning — Customer Churn Prediction Engine

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)]()
[![Progress](https://img.shields.io/badge/Progress-79%20%2F%20200%20Days-green.svg)](https://github.com/daystar-1nine/200DaysOfPython)

Part of the **200 Days of Python Challenge**.

---

## 🎯 Mission Overview
Day 79 advances from single-tree classification to **Ensemble Learning** with **Random Forest**. We build an end-to-end **Customer Churn Random Forest Prediction Engine** that combines Bootstrap Aggregation (Bagging) and Random Subspace Feature Selection to dramatically reduce variance and eliminate single-tree instability.

---

## 📂 Project Structure

```text
Day 79/
├── app/
│   ├── __init__.py
│   ├── config.py                     # Configuration & hyperparameters
│   ├── loader.py                     # Raw dataset ingestion
│   ├── cleaner.py                    # Deduplication & null imputation
│   ├── validator.py                  # Schema & data integrity checks
│   ├── feature_engineering.py        # Domain ratios & high-risk flags
│   ├── preprocessing.py              # ColumnTransformer (Tree vs Linear)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_baseline.py      # L2-penalized Logistic Regression
│   │   ├── decision_tree.py          # Single Decision Tree
│   │   └── random_forest.py          # 100-Tree Random Forest with OOB
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── cross_validation.py       # 5-Fold Stratified Cross-Validation
│   │   └── grid_search.py            # Systematic GridSearchCV pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # F1, Precision, Recall, ROC-AUC, Brier
│   │   ├── confusion_matrix.py       # Confusion matrix breakdown (TP, FP, FN, TN)
│   │   ├── roc.py                    # ROC curve coordinates & AUC
│   │   └── precision_recall.py       # PR curve coordinates & Average Precision
│   ├── feature_importance.py         # MDI Gini importance with tree std dev
│   ├── permutation_importance.py     # Out-of-sample test permutation importance
│   ├── threshold.py                  # Decision threshold sweeping (0.05 - 0.95)
│   ├── business_cost.py              # Asymmetric loss function (FP=300, FN=2000)
│   ├── risk_scoring.py               # CRM Risk Tiers: Low, Medium, High, Critical
│   ├── insights.py                   # Automated narrative executive synthesis
│   ├── visualizations.py             # 14 publication-grade analytical charts
│   ├── report.py                     # Markdown and executive report generation
│   └── main.py                       # Orchestrator pipeline
├── data/
│   ├── raw/customer_churn.csv        # 2,505 customer records (15 features)
│   └── processed/                    # train.csv, test.csv
├── output/
│   ├── charts/                       # 14 high-res visualization figures (PNG)
│   ├── metrics/                      # Model benchmarks, sweeps, risk CSVs
│   └── Day79_Final_Report.md         # Generated executive report
├── practice/                         # 4 manual algorithmic exercises
│   ├── 01_majority_voting.py
│   ├── 02_bootstrap_sampling.py
│   ├── 03_voting_calculation.py
│   └── 04_cost_calculation.py
├── coding_challenges/                # 6 advanced ML challenges
│   ├── challenge_01_easy_rf.py
│   ├── challenge_02_n_estimators_sweep.py
│   ├── challenge_03_gini_vs_entropy.py
│   ├── challenge_04_cv_comparison.py
│   ├── challenge_05_grid_search_rf.py
│   └── challenge_06_complete_comparison.py
├── tests/                            # 58 comprehensive unit tests
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_validator.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_tuning.py
│   ├── test_evaluation.py
│   ├── test_feature_importance.py
│   ├── test_threshold.py
│   ├── test_business_cost.py
│   └── test_risk_scoring.py
├── pytest.ini
├── Day79.md                          # In-depth Masterclass & 30 Interview Q&As
└── README.md
```

---

## 🏆 Model Performance Benchmark (Test Set)

| Model Architecture | Accuracy | Precision | Recall | Specificity | F1 Score | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.8360 | 0.6387 | 0.8852 | 0.8206 | 0.7421 | 0.9388 | 0.8407 |
| **Single Decision Tree (Day 78)** | 0.9720 | 0.9431 | 0.9484 | 0.9806 | 0.9457 | 0.9848 | 0.9469 |
| **Random Forest (100 Trees)** | **0.9680** | **0.9577** | **0.9174** | **0.9859** | **0.9370** | **0.9825** | **0.9681** |
| **Tuned Random Forest** | **0.9680** | **0.9577** | **0.9174** | **0.9859** | **0.9370** | **0.9790** | **0.9620** |

- **Random Forest Out-of-Bag (OOB) Accuracy**: `0.9590`
- **5-Fold Stratified CV F1 Score**: `0.9272 ± 0.0125` (lower variance and superior stability compared to single tree)

---

## 💰 Business Impact & Cost Optimization

In customer churn prediction, misclassification costs are asymmetric:
- **False Positive ($C_{FP} = \text{INR } 300$)**: Unnecessary retention marketing outreach.
- **False Negative ($C_{FN} = \text{INR } 2,000$)**: Lost customer lifetime revenue.

- **Default Threshold (0.50) Cost**: INR 16,700.00
- **Cost-Optimal Threshold (0.30) Cost**: INR 12,000.00
- **Net Business Value Unlocked**: **INR 4,700.00** (28.1% cost reduction)

---

## 📊 Visual Analytics (14 Charts)
Saved to `output/charts/`:
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

---

## 🚀 Quickstart & Execution

```bash
# 1. Run the end-to-end pipeline
python app/main.py

# 2. Run all unit tests (58 passing)
pytest tests -v

# 3. Run coding challenges
python coding_challenges/challenge_06_complete_comparison.py
```

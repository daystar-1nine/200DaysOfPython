# 🚀 End-to-End ML Model Selection Engine — Day 80 / 200

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/tests-65%20passed-brightgreen.svg)]()
[![Progress](https://img.shields.io/badge/Progress-80%20%2F%20200%20Days-green.svg)](https://github.com/daystar-1nine/200DaysOfPython)

Part of the **200 Days of Python Challenge** (Day 80 — **40.0% Complete Milestone** 🔥).

---

## 🎯 Objective
Determine the most defensible machine learning model for **Customer Churn Prediction** using a disciplined, production-grade ML workflow:
- **Statistical Evaluation**: Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, Average Precision (PR-AUC), and Brier Score.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation recording both mean and stability (standard deviation).
- **Hyperparameter Optimization**: Systematic `GridSearchCV` on training folds only.
- **Data Leakage Elimination**: Strict pipeline encapsulation via Scikit-Learn `ColumnTransformer`.
- **Financial Optimization**: Asymmetric business loss calibration ($C_{FP} = \text{INR } 300, C_{FN} = \text{INR } 2,000$).
- **Interpretability**: MDI Gini impurity reduction and test-set Permutation Importance.

---

## 📂 Project Architecture

```text
Day 80/
├── app/
│   ├── __init__.py
│   ├── config.py                     # AppConfig & business cost definitions
│   ├── main.py                       # End-to-end pipeline orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Safe dataset ingestion
│   │   ├── cleaner.py                # Deduplication & median/mode imputation
│   │   └── validator.py              # Schema & target integrity validation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── scaler.py                 # Numerical standardizer
│   │   ├── encoder.py                # Categorical one-hot encoder
│   │   └── pipeline.py               # ColumnTransformer & stratified splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py               # DummyClassifier (most frequent class)
│   │   ├── logistic.py               # L2-penalized Logistic Regression
│   │   ├── decision_tree.py          # Decision Tree Classifier
│   │   └── random_forest.py          # 100-Tree Random Forest with OOB
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Comprehensive classification metrics
│   │   ├── cross_validation.py       # 5-fold Stratified CV engine
│   │   ├── confusion_matrix.py       # Breakdown: TP, FP, FN, TN, specificity
│   │   ├── roc.py                    # ROC curve & AUC calculation
│   │   ├── precision_recall.py       # PR curve & Average Precision
│   │   └── calibration.py            # Calibration curves & Brier score
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── grids.py                  # Hyperparameter grids
│   │   └── search.py                 # GridSearchCV optimization
│   ├── threshold.py                  # Decision threshold sweeping (0.05 - 0.95)
│   ├── business_cost.py              # Financial loss matrix ($C_{FP}=300, C_{FN}=2000$)
│   ├── feature_importance.py         # MDI Gini importance with tree variance
│   ├── permutation_importance.py     # Unseen test permutation importance
│   ├── model_comparison.py           # Multi-model selection & ranking matrix
│   ├── visualizations.py             # 17 publication-grade analytical charts
│   ├── insights.py                   # Automated executive narrative synthesis
│   └── report.py                     # Milestone markdown report generator
├── data/
│   ├── raw/customer_churn.csv        # 2,505 customer records (15 features)
│   └── processed/                    # Cleaned dataset partition
├── output/
│   ├── data_quality_report.csv
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   ├── test_results.csv
│   ├── predictions.csv
│   ├── threshold_analysis.csv
│   ├── business_cost_analysis.csv
│   ├── feature_importance.csv
│   ├── permutation_importance.csv
│   ├── Day80_Final_Report.md
│   └── charts/                       # 17 analytical charts (PNG)
├── coding_challenges/                # 5 professional coding challenges
│   ├── challenge_01_generic_evaluator.py
│   ├── challenge_02_model_comparison.py
│   ├── challenge_03_threshold_optimizer.py
│   ├── challenge_04_data_leakage_detection.py
│   └── challenge_05_generic_model_report.py
├── tests/                            # 65 automated unit tests
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_validator.py
│   ├── test_preprocessing.py
│   ├── test_baseline.py
│   ├── test_logistic.py
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   ├── test_metrics.py
│   ├── test_cv.py
│   ├── test_tuning.py
│   ├── test_threshold.py
│   ├── test_business_cost.py
│   └── test_model_comparison.py
├── pytest.ini
├── requirements.txt
├── Day80.md                          # Masterclass documentation & 30 Interview Q&As
└── README.md
```

---

## 🏆 Model Selection Benchmark Matrix

| Model Architecture | 5-Fold CV ROC-AUC | 5-Fold CV AP | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test AP | Test Business Cost |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Dummy)** | 0.5000 (±0.000) | 0.2520 | 0.7480 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2520 | INR 252,000 |
| **Logistic Regression** | 0.9431 (±0.005) | 0.8523 | 0.8400 | 0.6480 | 0.8889 | 0.7492 | 0.9409 | 0.8465 | INR 37,300 |
| **Random Forest** | 0.9816 (±0.004) | 0.9575 | 0.9740 | 0.9593 | 0.9365 | 0.9494 | 0.9843 | 0.9634 | INR 10,700 |
| **Decision Tree (Tuned)** | **0.9771 (±0.009)** | **0.9351** | **0.9680** | **0.9044** | **0.9762** | **0.9389** | **0.9897** | **0.9516** | **INR 9,900** |

---

## 💰 Business Optimization
- **False Positive Cost ($C_{FP} = \text{INR } 300$)**: Unneeded retention outreach.
- **False Negative Cost ($C_{FN} = \text{INR } 2,000$)**: Unintercepted churn revenue loss.
- **Champion Model**: The Tuned Decision Tree achieved the lowest financial loss (**INR 9,900**) and highest recall (**97.6%**), followed closely by the Random Forest (**INR 10,700**).
- **Value Generated**: The champion model unlocked **INR 242,100** in avoided churn losses compared to the unmanaged baseline.

---

## 📊 Visual Analytics (17 Figures)
Saved in `output/charts/`:
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

---

## 🛡️ Data Leakage Prevention Checklist
- [x] Train-test split (80/20) executed prior to any feature transformation.
- [x] Preprocessing estimators fitted strictly on training data (`ColumnTransformer`).
- [x] Test partition never touched during cross-validation or hyperparameter search.
- [x] Target and future variables audited and excluded from feature sets.
- [x] Final test set evaluated exactly once after champion selection.

---

## 🧪 Testing & Execution

```bash
# 1. Run full end-to-end model selection pipeline
python app/main.py

# 2. Execute full automated test suite (65 passing tests)
pytest tests -v

# 3. Run coding challenges
python coding_challenges/challenge_04_data_leakage_detection.py
```

---

## 📌 Documented Limitations & Production Assumptions
1. **Dataset Volume**: 2,500 rows provide solid prototype signals, but production retraining on 100k+ enterprise records is recommended.
2. **Feature Lineage**: Assumes payment failures and support ticket logs are updated synchronously in real time.
3. **Drift Monitoring**: Post-deployment tracking is necessary to monitor covariate shift in contract lengths and tenure distributions.

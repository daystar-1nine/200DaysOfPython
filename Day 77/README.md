# Day 77: Intelligent Customer Risk Classification Engine

## 🚀 Overview
Day 77 of the **200 Days of Python** challenge implements an enterprise-grade **Intelligent Customer Risk Classification Engine**. This platform moves beyond simple accuracy metrics to address multi-class classification, severe class imbalance, stratified cross-validation, precision-recall optimization, probability calibration, and business cost minimization.

---

## 🎯 Dual Classification Tasks
1. **Binary Classification (Customer Churn)**:
   - Target: `Churn` ($0 = \text{Retained}$, $1 = \text{Churned}$)
   - Models: Logistic Regression, Balanced Logistic Regression, Random Forest Classifier
2. **Multi-Class Classification (Customer Risk Level)**:
   - Target: `Risk_Level` (`Low`, `Medium`, `High`)
   - Architecture: One-vs-Rest (OvR) multi-class decomposition

---

## 📊 Key Results & Model Performance

### Binary Churn Prediction
| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC ($AP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 91.5% | 88.1% | 88.1% | 0.881 | 0.980 | 0.965 |
| **Balanced Logistic** | **91.5%** | 84.5% | **93.2%** | **0.886** | **0.979** | **0.964** |
| **Random Forest** | 88.4% | 89.4% | 76.7% | 0.826 | 0.953 | 0.932 |

### Multi-Class Risk Classification
| Model | Accuracy | Macro $F_1$ | Weighted $F_1$ | ROC-AUC (OvR Macro) | High Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic OvR** | 80.3% | 0.767 | 0.790 | 0.936 | 80.8% |
| **Balanced Logistic OvR** | **85.2%** | **0.828** | **0.849** | **0.937** | **81.7%** |
| **Random Forest** | 78.5% | 0.747 | 0.777 | 0.909 | 65.8% |

---

## 💰 Business Cost Optimization
- **False Negative Cost ($C_{FN}$)**: INR 5,000 (Lost customer lifetime value)
- **False Positive Cost ($C_{FP}$)**: INR 300 (Retention discount expenditure)
- **Default Baseline Threshold ($\\tau = 0.50$) Cost**: INR 69,000.00
- **Cost-Optimized Threshold ($\\tau = 0.15$) Cost**: INR 21,300.00
- **Net Projected Financial Savings**: **INR 47,700.00 (69.1% cost reduction)**

---

## 📈 Generated Visualizations (16 Charts)
All charts are saved in `output/charts/`:
1. `churn_distribution.png` — Binary churn proportion barplot
2. `risk_distribution.png` — Multi-class risk level distribution
3. `age_vs_churn.png` — Age distribution KDE by churn status
4. `charges_vs_churn.png` — Monthly charges boxplot
5. `tenure_vs_churn.png` — Customer tenure violinplot
6. `contract_churn.png` — Churn percentage by contract type
7. `binary_confusion_matrix.png` — 2x2 annotated heatmap
8. `multiclass_confusion_matrix.png` — 3x3 normalized heatmap
9. `roc_curve.png` — Binary ROC with Youden's $J$ + Multi-class OvR curves
10. `precision_recall_curve.png` — Binary PR curve + Multi-class PR curves
11. `threshold_metrics.png` — Precision, Recall, and Accuracy vs. threshold
12. `threshold_f1.png` — $F_1$ curve and business cost curve vs. threshold
13. `probability_distribution.png` — Predicted probabilities histogram by true class
14. `calibration_curve.png` — Reliability calibration diagram and Brier score
15. `feature_importance.png` — Top coefficients and predictive feature weights
16. `class_performance.png` — Per-class Precision, Recall, and $F_1$ comparison

---

## 🧪 Testing & Execution

### Run Full Pipeline
```bash
python app/main.py
```

### Run Unit Tests
```bash
pytest tests -v
```
**Result**: 54 passed in 1.29s.

### Run Practice Tasks & Coding Challenges
```bash
python practice/task1_multiclass_logistic.py
python coding_challenges/challenge1_minority_champion.py
```

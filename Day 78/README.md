# Day 78: Customer Churn Decision Tree Engine

## 🚀 Overview
Day 78 of the **200 Days of Python** challenge implements an interpretable **Customer Churn Decision Tree Engine**. Moving beyond the linear equations of Logistic Regression (Days 76–77), this engine learns non-linear decision rules and interaction patterns to predict subscription cancellations while exposing human-readable inference logic.

---

## 🎯 Key Architectural Highlights
1. **Decision Tree Architecture**:
   - `DecisionTreeClassifier` with both Gini Impurity and Shannon Entropy splitting criteria.
   - Scale-invariant preprocessing pipeline (`passthrough` numeric features + `OneHotEncoder` for categoricals).
2. **Hyperparameter Tuning & Regularization**:
   - 5-fold Stratified Grid Search over `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion`.
   - Depth-curve bias-variance diagnostic tracking train vs. test $F_1$ across depths 1 to 15.
3. **Interpretability & Explainability Suite**:
   - Mean Decrease in Impurity (MDI) feature importance.
   - Permutation importance ($F_1$-score drop across test shuffling).
   - Global decision rule extraction via `export_text`.
   - Customer-level step-by-step decision path tracing.
4. **Business Cost Optimization**:
   - Evaluates financial risk costs ($FP = \text{INR } 300$, $FN = \text{INR } 5,000$).
   - Replaces the arbitrary $\tau = 0.50$ threshold with a cost-minimizing threshold.

---

## 📊 Model Comparison & Results

Evaluated on an untouched test set of 495 customers (`output/model_metrics.csv`):

| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC ($AP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** | 89.3% | 79.2% | 78.6% | 0.789 | 0.945 | 0.848 |
| **Decision Tree (Gini Unconstrained)** | 93.9% | 88.7% | 87.3% | 0.880 | 0.918 | 0.807 |
| **Decision Tree (Entropy Unconstrained)** | 94.9% | 90.4% | 89.7% | 0.900 | 0.932 | 0.837 |
| **Tuned Decision Tree** | **96.0%** | **90.8%** | **93.7%** | **0.922** | **0.973** | **0.933** |

### Optimal Hyperparameters (via 5-Fold Stratified CV):
- `criterion`: **gini**
- `max_depth`: **6**
- `min_samples_leaf`: **5**
- `min_samples_split`: **2**
- Best CV $F_1$-Score: **0.9330**

### Financial Cost Optimization:
- Default Threshold ($\tau = 0.50$) Cost: **INR 43,600.00**
- Optimal Cost Threshold ($\tau = 0.25$) Cost: **INR 28,600.00**
- **Net Projected Financial Savings: INR 15,000.00 (34.4% cost reduction)**

---

## 📈 Generated Visualizations (12 Charts)
Saved in `output/charts/`:
1. `churn_distribution.png` — Baseline binary churn proportion.
2. `contract_churn.png` — Churn rate broken down by contract type.
3. `internet_churn.png` — Churn rate by internet service provider.
4. `charges_churn.png` — Monthly charges distribution boxplot.
5. `tenure_churn.png` — Customer tenure distribution violinplot.
6. `decision_tree.png` — Pruned tree diagram generated with `plot_tree`.
7. `feature_importance.png` — Top 10 features by Mean Decrease in Impurity.
8. `permutation_importance.png` — Top 10 features by permutation score drop.
9. `confusion_matrix.png` — Heatmap of test confusion matrix.
10. `roc_curve.png` — Receiver Operating Characteristic curve.
11. `precision_recall_curve.png` — Precision-Recall curve with Average Precision.
12. `depth_vs_score.png` — Train vs. Test $F_1$ across tree depths (1–15).

---

## 🧪 Testing & Execution

### Run Main Pipeline
```bash
python app/main.py
```

### Run Unit Tests
```bash
pytest tests -v
```
**Result**: 47 passed in 3.31s.

### Run Practice Tasks & Coding Challenges
```bash
python practice/task1_manual_decision_tree.py
python coding_challenges/challenge4_explain_customer.py
```

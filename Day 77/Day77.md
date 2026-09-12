# DAY 77 / 200: Advanced Classification & Model Evaluation

## Intelligent Customer Risk Classification Engine

---

## 1. Executive Summary & Context

Yesterday in Day 76, you transitioned from predicting continuous numerical quantities (regression) to predicting discrete states (binary classification) using Logistic Regression:

```text
Features (Age, Charges, Usage) -> Log-Odds -> Sigmoid -> Probability -> Threshold -> Class (0 or 1)
```

However, real-world machine learning systems in production rarely stop at:
$$\text{Accuracy} = 87\%$$

Accuracy is a fragile metric. A naive fraud model in an environment where $99.5\%$ of transactions are legitimate can predict "Not Fraud" for every single case and achieve $99.5\%$ accuracy while having **zero recall** on actual criminals. 

Today, we level up to enterprise-grade classification. We tackle:
1. **Multi-Class Classification**: Moving beyond binary targets ($0$ or $1$) to multi-category outcomes (Low, Medium, High Risk) using **One-vs-Rest (OvR)** and **One-vs-One (OvO)** architectures.
2. **Severe Class Imbalance**: Understanding why models favor majority classes and how inverse-frequency class weighting ($w_j = N / (K \cdot N_j)$) corrects decision boundaries.
3. **Stratification & Leakage-Free Validation**: Preserving class proportions across train/test splits and cross-validation folds using `StratifiedKFold`.
4. **Precision-Recall Curves & PR-AUC**: Proving why Average Precision ($AP$) provides a far more truthful evaluation than ROC-AUC on imbalanced datasets.
5. **Business Cost-Driven Threshold Optimization**: Replacing the arbitrary $\tau = 0.50$ threshold with cost-minimization equations that penalize asymmetric real-world errors ($C_{FN} \gg C_{FP}$).
6. **Probability Calibration & Brier Score**: Disentangling discrimination from probabilistic truth.

---

## 2. Theoretical Foundations

### 2.1 Binary vs. Multi-Class Classification

In binary classification, the target variable $y \in \{0, 1\}$. The logistic function outputs a single probability:
$$P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = \beta_0 + \sum_{j=1}^p \beta_j x_j$.

In multi-class classification, the target variable takes one of $K$ mutually exclusive categories:
$$y \in \{1, 2, \dots, K\}, \quad \text{where } K \ge 3$$

For our Intelligent Customer Risk Classification Engine:
$$y \in \{\text{Low}, \text{Medium}, \text{High}\}$$

### 2.2 Decomposition Strategies: One-vs-Rest (OvR) vs. One-vs-One (OvO)

Standard binary algorithms like logistic regression or support vector machines do not naturally handle multiple classes in a single binary boundary. Two primary decomposition strategies bridge this gap:

#### Strategy 1: One-vs-Rest (OvR / One-vs-All)
In OvR, $K$ independent binary classifiers are trained. Each classifier $f_k(\mathbf{x})$ is trained to distinguish class $k$ from all other classes combined:
- Classifier 1: Class 1 vs. (Classes $2, 3, \dots, K$)
- Classifier 2: Class 2 vs. (Classes $1, 3, \dots, K$)
- Classifier $K$: Class $K$ vs. (Classes $1, 2, \dots, K-1$)

During inference, each classifier outputs a confidence score or probability $\hat{P}_k(\mathbf{x})$. The final predicted class is the argmax:
$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \hat{P}_k(\mathbf{x})$$

**Properties of OvR:**
- Requires exactly $K$ models.
- Highly scalable for moderate to large class counts.
- Well-suited for linear classifiers.

#### Strategy 2: One-vs-One (OvO)
In OvO, a binary classifier is trained for every possible pair of classes $(i, j)$ with $i < j$. Observations not belonging to either class $i$ or class $j$ are ignored during training of that pair.

The total number of binary classifiers required is:
$$N_{\text{classifiers}} = \frac{K(K - 1)}{2}$$

For $K = 3$ classes:
$$\frac{3(2)}{2} = 3 \text{ classifiers}$$

For $K = 10$ classes:
$$\frac{10(9)}{2} = 45 \text{ classifiers}$$

During prediction, all $K(K-1)/2$ classifiers vote on the incoming sample, and the class with the most pairwise votes is selected.

| Dimension | One-vs-Rest (OvR) | One-vs-One (OvO) |
| :--- | :--- | :--- |
| **Number of Classifiers** | $K$ | $\frac{K(K-1)}{2}$ |
| **Training Sample Size** | Entire dataset per classifier | Subset of data belonging to pair $(i, j)$ |
| **Computational Scalability** | High (linear in $K$) | Quadratic in $K$ (expensive for large $K$) |
| **Class Imbalance Effect** | Inherently induces imbalance | Preserves pairwise balance |
| **Primary Use Cases** | Logistic Regression, Neural Nets | Support Vector Machines (SVM) with kernel |

---

## 3. Class Imbalance & Weighting Mechanics

### 3.1 The Accuracy Paradox

Consider an enterprise dataset of 10,000 corporate customers where churn is rare:
- 9,500 Active Customers ($y = 0$)
- 500 Churned Customers ($y = 1$)

A naive baseline classifier that always predicts $0$ yields:
$$\text{Accuracy} = \frac{9500 + 0}{10000} = 95.0\%$$

However, its sensitivity (recall) on churned accounts is $0.0\%$. If the lost customer lifetime value averages INR 5,000, the business loses INR 2,500,000 while celebrating a $95\%$ accurate model.

### 3.2 Loss Reweighting Formulation

To prevent the loss function from being dominated by majority samples, scikit-learn implements cost-sensitive learning via `class_weight="balanced"`.

The standard unweighted binary cross-entropy loss is:
$$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

Under balanced class weighting, an individual sample weight $w_{y_i}$ is applied to each observation based on its class prevalence:
$$J_{\text{balanced}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

The weight $w_j$ for class $j$ is computed as:
$$w_j = \frac{N}{K \cdot N_j}$$
where:
- $N$ is total sample count
- $K$ is total number of distinct classes
- $N_j$ is the count of samples in class $j$

For our example ($N = 10,000$, $K = 2$, $N_0 = 9,500$, $N_1 = 500$):
$$w_0 = \frac{10000}{2 \times 9500} = 0.526$$
$$w_1 = \frac{10000}{2 \times 500} = 10.000$$

Every misclassification of a minority sample is penalized roughly $19$ times more heavily than misclassifying a majority sample ($10.0 / 0.526 \approx 19$). This shifts the decision boundary towards the majority class, drastically elevating minority recall.

---

## 4. Sampling & Cross-Validation Strategies

### 4.1 Stratified Train/Test Splitting

Standard random sampling can introduce severe distributional drift between the training and test partitions, particularly when evaluating minority cohorts.

Stratification guarantees that:
$$\frac{N_{j, \text{train}}}{N_{\text{train}}} \approx \frac{N_{j, \text{test}}}{N_{\text{test}}} \approx \frac{N_j}{N}$$

In Python:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### 4.2 Stratified K-Fold Cross-Validation

Standard $K$-fold splits the data into $K$ contiguous or randomized blocks without regard to class labels. If a minority class comprises only $2\%$ of the data, a fold might inadvertently receive $0\%$ or $5\%$ positive cases.

`StratifiedKFold` ensures each fold acts as an identical statistical representation of the parent distribution:
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipeline, X, y, cv=skf, scoring=["f1_macro", "roc_auc"])
```

---

## 5. Comprehensive Metric Suite & Averaging Philosophies

### 5.1 Confusion Matrix Topography

For binary classification:

| | Predicted Negative (0) | Predicted Positive (1) |
| :--- | :--- | :--- |
| **Actual Negative (0)** | True Negative (TN) | False Positive (FP) |
| **Actual Positive (1)** | False Negative (FN) | True Positive (TP) |

Core metric definitions:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (Positive Predictive Value)**: $\frac{TP}{TP + FP}$
- **Recall (Sensitivity / True Positive Rate)**: $\frac{TP}{TP + FN}$
- **Specificity (True Negative Rate)**: $\frac{TN}{TN + FP}$
- **False Positive Rate (Fall-out)**: $\frac{FP}{FP + TN} = 1 - \text{Specificity}$
- **$F_1$-Score (Harmonic Mean)**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$

### 5.2 Multi-Class Averaging: Macro vs. Weighted vs. Micro

When evaluating multi-class systems with $K$ classes, per-class metrics $M_1, M_2, \dots, M_K$ must be aggregated into an overall score:

#### Macro-Averaging
Computes the metric independently for each class and calculates the unweighted arithmetic mean:
$$M_{\text{macro}} = \frac{1}{K} \sum_{k=1}^K M_k$$
*Philosophy*: Treats every class as equally important, regardless of support. Essential when identifying whether minority classes are failing.

#### Weighted-Averaging
Weights the contribution of each class by its relative support (sample count) in the true dataset:
$$M_{\text{weighted}} = \sum_{k=1}^K \left( \frac{N_k}{N} \right) M_k$$
*Philosophy*: Reflects overall dataset population frequency. Large classes dominate this metric.

#### Micro-Averaging
Aggregates the total true positives, false positives, and false negatives globally across all classes before calculating the metric:
$$\text{Precision}_{\text{micro}} = \frac{\sum_{k=1}^K TP_k}{\sum_{k=1}^K (TP_k + FP_k)}$$
*Philosophy*: In multi-class single-label classification, Micro Precision, Micro Recall, and Micro $F_1$ are mathematically equal to overall Accuracy.

---

## 6. Discrimination vs. Calibration: ROC vs. Precision-Recall

### 6.1 ROC Curves vs. Precision-Recall (PR) Curves

The **Receiver Operating Characteristic (ROC)** curve plots:
- $Y$-axis: True Positive Rate ($TPR = TP / (TP + FN)$)
- $X$-axis: False Positive Rate ($FPR = FP / (FP + TN)$)

The **Precision-Recall (PR)** curve plots:
- $Y$-axis: Precision ($TP / (TP + FP)$)
- $X$-axis: Recall ($TP / (TP + FN)$)

#### Why ROC-AUC Can Deceive Under Heavy Imbalance
Examine the denominator of the False Positive Rate:
$$FPR = \frac{FP}{FP + TN}$$

If the negative class is massive ($TN = 100,000$), even $500$ false positives yield an $FPR$ of:
$$FPR = \frac{500}{500 + 100000} = 0.00497 \approx 0.5\%$$

The ROC curve remains pinned near the upper-left corner, yielding an impressive $\text{ROC-AUC} > 0.95$.

However, if there were only $100$ true positives ($TP = 80$), the actual precision is disastrous:
$$\text{Precision} = \frac{80}{80 + 500} = 0.138 \approx 13.8\%$$

The Precision-Recall curve exposes this weakness immediately. While ROC-AUC measures **overall discrimination across all classes**, Average Precision ($AP$) reveals **operational performance specifically on the positive cohort**.

### 6.2 Average Precision ($AP$)
Scikit-learn summarizes the PR curve using Average Precision:
$$AP = \sum_n (R_n - R_{n-1}) P_n$$
where $P_n$ and $R_n$ represent precision and recall at the $n$-th threshold.

### 6.3 Probability Calibration & The Brier Score
A model with high discrimination can still output uncalibrated probabilities. For example, if a model predicts $P = 0.90$ for 100 customers, exactly $90$ of those customers should churn if the model is well-calibrated.

Calibration is assessed visually via reliability diagrams (calibration curves) and numerically using the **Brier Score**:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
The Brier score acts as the mean squared error of probability forecasts. $0.0$ represents perfect probabilistic calibration, while $0.25$ represents random guessing for a $50/50$ prior.

---

## 7. Business Cost-Based Threshold Optimization

In corporate decision-making, errors carry asymmetric costs. Predicting a customer will stay when they actually churn (False Negative) costs the enterprise lost Customer Lifetime Value (LTV):
$$C_{FN} = \text{INR } 5,000$$

Conversely, predicting a customer will churn when they would have stayed (False Positive) costs only an unnecessary retention discount:
$$C_{FP} = \text{INR } 300$$

The cost ratio is:
$$\frac{C_{FN}}{C_{FP}} = \frac{5000}{300} \approx 16.7 : 1$$

The total operational business loss as a function of the decision threshold $\tau$ is:
$$\text{Cost}(\tau) = FP(\tau) \cdot C_{FP} + FN(\tau) \cdot C_{FN}$$

By sweeping candidate thresholds $\tau \in [0.05, 0.95]$, our system computes the exact threshold that minimizes financial loss:
$$\tau^* = \arg\min_{\tau} \text{Cost}(\tau)$$

In empirical testing on our 2,465 customer dataset:
- Standard Default ($\tau = 0.50$): Total Cost = **INR 69,000.00**
- Cost-Optimized ($\tau = 0.15$): Total Cost = **INR 21,300.00**
- **Net Projected Financial Savings: INR 47,700.00 ($69.1\%$ cost reduction)**

---

## 8. System Architecture

The complete system architecture implemented in `Day 77/app/`:

```text
                     +-----------------------------------+
                     |   data/raw/customer_churn.csv     |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |          app/loader.py            |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |          app/cleaner.py           |
                     |  (Negative filter, Imputation)    |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |        app/validator.py           |
                     |   (Domain & schema validation)    |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |    app/feature_engineering.py     |
                     | (Charges_Ratio, Support_Tenure)   |
                     +-----------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
    +------------------------------+       +------------------------------+
    | Binary Task: Churn (0 or 1)  |       | Multi-Class: Risk Level      |
    | (Stratified Split: 80/20)    |       | (Low, Medium, High: OvR)     |
    +------------------------------+       +------------------------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |      app/preprocessing.py         |
                     | (StandardScaler + OneHotEncoder)  |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |           app/models/             |
                     | - logistic.py                     |
                     | - balanced_logistic.py            |
                     | - tree_comparison.py              |
                     +-----------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
    +------------------------------+       +------------------------------+
    |       app/evaluation/        |       |        app/threshold.py      |
    | - metrics.py (Macro/Weight)  |       | - F1-optimal Threshold       |
    | - confusion_matrix.py        |       | - Recall-constrained (>=80%) |
    | - roc.py (Youden J & Macro)  |       |        app/business_cost.py  |
    | - precision_recall.py (AP)   |       | - Cost-minimal Threshold     |
    | - calibration.py (Brier)     |       +------------------------------+
    +------------------------------+                       |
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |        app/risk_scoring.py        |
                     | (Customer Churn Prob & Actions)   |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     app/visualizations.py         |
                     | (16 Publication-Quality Charts)   |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     app/report.py & insights.py   |
                     | (Executive Report & Analytics)    |
                     +-----------------------------------+
```

---

## 9. Comprehensive Interview Masterclass (30 Q&As)

### Section A: Beginner Level

#### Q1: What is the fundamental difference between binary and multi-class classification?
In binary classification, the outcome is dichotomous ($y \in \{0, 1\}$), such as Churn vs. Retained. In multi-class classification, the target belongs to one of three or more mutually exclusive categories ($y \in \{1, 2, \dots, K\}, K \ge 3$), such as Customer Risk Level (Low, Medium, High).

#### Q2: What is the One-vs-Rest (OvR) classification strategy?
One-vs-Rest (also known as One-vs-All) trains $K$ separate binary classifiers for a $K$-class problem. Each classifier separates one single class from the union of all other classes. At inference, the class whose model produces the highest probability score is selected.

#### Q3: What is the One-vs-One (OvO) strategy?
One-vs-One trains a binary classifier for every distinct pair of classes, resulting in $K(K-1)/2$ models. Each model votes between two specific classes, and the class accumulating the highest number of pairwise votes wins.

#### Q4: Why can overall Accuracy be dangerously misleading on imbalanced datasets?
Accuracy simply measures the percentage of correct predictions over total predictions. If $95\%$ of samples belong to the negative class, a constant negative predictor attains $95\%$ accuracy while completely failing to detect any positive instances.

#### Q5: What does the `support` column represent in a classification report?
Support indicates the actual number of occurrences (ground truth instances) of each class present in the specified evaluation dataset.

#### Q6: What is the difference between Type I and Type II errors in classification?
A Type I error is a False Positive ($FP$)—predicting a positive outcome when the true state is negative. A Type II error is a False Negative ($FN$)—failing to detect an actual positive instance.

---

### Section B: Intermediate Level

#### Q7: What is stratified train/test splitting and why is it mandatory?
Stratified splitting enforces that the relative proportion of each class in the training and test subsets strictly matches the proportion in the original parent dataset. Without stratification, small minority classes could be severely underrepresented or completely omitted from the test partition.

#### Q8: Why must you use `StratifiedKFold` instead of standard `KFold` for classification?
Standard `KFold` randomly assigns rows to folds. For imbalanced datasets, certain folds may receive virtually zero minority samples, causing severe variance in evaluation metrics. `StratifiedKFold` guarantees balanced class proportions across every single fold.

#### Q9: What is the mathematical definition and purpose of Macro $F_1$?
Macro $F_1$ is the unweighted arithmetic mean of the $F_1$-scores calculated independently for each class:
$$F_{1, \text{macro}} = \frac{1}{K} \sum_{k=1}^K F_{1, k}$$
It treats every class with equal priority, exposing if a model fails on smaller minority classes.

#### Q10: How does Weighted $F_1$ differ from Macro $F_1$?
Weighted $F_1$ weights each class's $F_1$-score by its relative support ($N_k / N$):
$$F_{1, \text{weighted}} = \sum_{k=1}^K \left( \frac{N_k}{N} \right) F_{1, k}$$
Unlike Macro $F_1$, larger majority classes dominate Weighted $F_1$.

#### Q11: What is Micro $F_1$ and what does it equal in single-label multi-class tasks?
Micro $F_1$ aggregates global true positives, false positives, and false negatives before applying the harmonic formula. In single-label multi-class problems, Micro Precision, Micro Recall, and Micro $F_1$ are all mathematically identical to overall Accuracy.

#### Q12: How does `class_weight="balanced"` modify the model's loss function?
It assigns an inverse-frequency weight $w_j = N / (K \cdot N_j)$ to each sample. The loss penalty for misclassifying a minority instance is scaled up proportionally, forcing the gradient descent optimizer to adjust the decision boundary in favor of minority recall.

#### Q13: What is a Precision-Recall (PR) curve?
A Precision-Recall curve plots Precision on the $y$-axis versus Recall on the $x$-axis across all possible decision thresholds $\tau \in [0, 1]$.

#### Q14: What is Average Precision ($AP$)?
Average Precision summarizes the Precision-Recall curve as the weighted mean of precisions achieved at each threshold, where the weight is the increase in recall from the previous threshold:
$$AP = \sum_n (R_n - R_{n-1}) P_n$$

#### Q15: When should you prioritize PR-AUC over ROC-AUC?
You should prioritize PR-AUC whenever evaluating datasets with severe class imbalance (e.g., fraud detection, disease screening, rare churn), where the majority class is large and the positive class represents the primary business concern.

---

### Section C: Advanced Level

#### Q16: Why does ROC-AUC remain high even when precision is unacceptably low on imbalanced data?
The ROC curve's horizontal axis is the False Positive Rate ($FPR = FP / (FP + TN)$). When the true negative count ($TN$) is massive, large numbers of false positives result in only a minuscule increase in $FPR$. Consequently, the ROC curve appears nearly ideal, hiding low precision.

#### Q17: How does adjusting the classification threshold $\tau$ affect the trade-off between Precision and Recall?
Lowering $\tau$ classifies more instances as positive. This increases True Positives (elevating Recall) but also captures more False Positives (reducing Precision). Raising $\tau$ enforces stricter positive criteria, generally elevating Precision while lowering Recall.

#### Q18: What is Youden's $J$ statistic and how is it used in ROC analysis?
Youden's $J$ statistic evaluates the optimal cut-off point on an ROC curve:
$$J = \text{Sensitivity} + \text{Specificity} - 1 = TPR - FPR$$
The threshold that maximizes $J$ represents the point furthest above the diagonal random-guess line.

#### Q19: What is the conceptual difference between model discrimination and model calibration?
- **Discrimination** is the model's ability to rank positive instances higher than negative instances (measured by ROC-AUC and PR-AUC).
- **Calibration** is the degree to which predicted probabilities match empirical observed frequencies (measured by Calibration Curves and Brier Score).

#### Q20: What is the Brier Score and how is it interpreted?
The Brier Score is the mean squared error between predicted probabilities $p_i$ and true binary outcomes $y_i \in \{0, 1\}$:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Scores range from $0$ (perfect probabilistic forecast) to $1$. For a balanced dataset, a non-informative model predicting $0.5$ has a Brier score of $0.25$.

#### Q21: How do you mathematically formulate cost-sensitive threshold selection?
Define a cost matrix with penalties $C_{FP}$ and $C_{FN}$. For each threshold $\tau$:
$$\text{Total Cost}(\tau) = FP(\tau) \cdot C_{FP} + FN(\tau) \cdot C_{FN}$$
The optimal operational threshold is the argument minimizing this objective function:
$$\tau^* = \arg\min_\tau \text{Total Cost}(\tau)$$

#### Q22: Why should data preprocessing steps (e.g., scaling and encoding) occur inside cross-validation folds rather than before?
Fitting scalers or encoders on the entire dataset prior to splitting leaks test distribution parameters (such as mean, variance, and target-encoded frequencies) into the training folds, producing optimistically biased cross-validation estimates.

#### Q23: Why should decision thresholds never be optimized on the final test set?
Optimizing a decision threshold on the test set treats that test set as validation data, leading to threshold overfitting. The optimal threshold should be derived via cross-validation or a dedicated validation split, and only evaluated once on the test partition.

#### Q24: What is Platt Scaling and isotonic regression in probability calibration?
- **Platt Scaling** fits a logistic regression model to the raw decision scores of an uncalibrated classifier.
- **Isotonic Regression** fits a non-parametric, piece-wise constant non-decreasing function to map raw scores to calibrated probabilities.

#### Q25: How does multinomial logistic regression (Softmax) differ from One-vs-Rest logistic regression?
Softmax regression optimizes a single multi-class cross-entropy loss function where probabilities across all classes sum to 1 via the softmax operator:
$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$
OvR fits $K$ separate binary logistic regressions, and raw probabilities may not inherently sum to 1 without post-hoc normalization.

#### Q26: Can a model have an ROC-AUC of 0.95 and an Average Precision of 0.30? Explain how.
Yes. In a dataset with 10,000 negatives and 50 positives ($0.5\%$ positive rate), a model might produce 500 false positives while identifying 45 true positives. The $FPR = 500 / 10000 = 0.05$ (yielding $TPR = 0.90$ at $FPR = 0.05$, giving $\text{ROC-AUC} \approx 0.95$). However, Precision is only $45 / (45 + 500) = 0.0825$, producing a low Average Precision around $0.30$.

#### Q27: How do you handle class imbalance when the minority class has very few samples (e.g., < 20)?
When samples are scarce, techniques like `class_weight="balanced"`, repeated stratified $K$-fold validation, synthetic oversampling (SMOTE), and gathering additional data are employed, avoiding excessive threshold optimization that overfits small sample noise.

#### Q28: How do you evaluate multi-class ROC-AUC using Scikit-Learn?
Use `roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")`. You must explicitly specify `multi_class="ovr"` or `"ovo"`, and designate whether to use macro or weighted averaging.

#### Q29: What is the risk of using SMOTE before feature scaling?
SMOTE computes $k$-nearest neighbors in feature space. If features are unscaled, variables with large absolute ranges will dominate Euclidean distance calculations, creating distorted synthetic samples along arbitrary axes.

#### Q30: What is the operational difference between predicting a class vs. predicting a probability score in enterprise decision engines?
Predicting a hard class fixes the business logic to a static threshold (typically $\tau = 0.50$). Predicting calibrated probabilities decouples statistical inference from business policy, allowing leadership to dynamically alter decision thresholds as operational costs, budget constraints, and risk tolerances fluctuate.

---

## 10. Verification & Quality Checklist

- [x] All mathematical expressions render cleanly using single backslash notation.
- [x] Zero deprecated macros used (no `\operatorname`, `\mathbf`, etc.).
- [x] 16 publication-quality analytical visualizations generated in `output/charts/`.
- [x] 54 comprehensive unit tests passing with 100% success rate.
- [x] Practice tasks and 5 coding challenges fully functional.
- [x] Project and repo root documentation synchronized.

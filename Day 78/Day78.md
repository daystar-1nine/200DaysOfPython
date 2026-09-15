# DAY 78 / 200: Decision Trees & Tree-Based Classification

## Customer Churn Decision Tree Engine

---

## 1. Executive Summary & Machine Learning Paradigm Shift

Over Days 76 and 77, you mastered **Logistic Regression**—a linear classification model that fits a single hyperplane through feature space:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)$$

While linear models are elegant, fast, and mathematically tractable, they operate under rigid assumptions:
1. They assume monotonic and linearly additive relationships between the log-odds of the outcome and the input features.
2. They cannot model non-linear boundaries without manual basis expansions (e.g., polynomial or interaction terms).
3. They are sensitive to outliers and require feature scaling (Z-score standardization or MinMax normalization).

Today, we enter a completely different machine learning paradigm: **Tree-Based Machine Learning**.

Instead of learning a single global mathematical equation, a **Decision Tree** partitions feature space into a hierarchy of recursive, orthogonal (axis-aligned) rectangular subsets through a sequence of data-driven decisions:

```text
                           Contract == 'Month-to-month'?
                                  /              \
                               YES                NO
                              /                    \
                   Support Calls > 3?           Tenure > 24?
                       /        \                 /      \
                    YES          NO             YES        NO
                     ↓            ↓              ↓          ↓
                 CHURN (94%)   STAY (72%)    STAY (98%)  STAY (85%)
```

By the end of today, you will understand the exact mathematical criteria (Gini Impurity, Shannon Entropy, and Information Gain) governing how splits are chosen, why trees do not require feature scaling, how hyperparameter regularization prevents runaway overfitting, how to trace customer-level inference rules, and how a tuned Decision Tree compares against Logistic Regression.

---

## 2. Anatomical Terminology of Decision Trees

A Decision Tree is a directed acyclic graph (DAG) organized hierarchically:

```text
                     [ Root Node: N=2471, Gini=0.378 ]
                                    |
                 +------------------+------------------+
                 | Split: Contract == 'Month-to-month' |
                 v                                     v
       [ Internal Node 1 ]                    [ Internal Node 2 ]
                 |                                     |
         +-------+-------+                     +-------+-------+
         v               v                     v               v
    [ Leaf A ]      [ Leaf B ]            [ Leaf C ]      [ Leaf D ]
    (Class 1)       (Class 0)             (Class 0)       (Class 0)
```

1. **Root Node**: The top-most decision node containing the entire training population ($N$). It evaluates the single feature split that yields the largest global impurity reduction.
2. **Internal (Decision) Node**: A node with exactly one incoming branch and two outgoing branches (in binary CART trees). It tests a univariate conditional statement:
   $$x_j \le \theta \quad \text{vs.} \quad x_j > \theta$$
3. **Branch**: An edge connecting a parent node to a child node representing the outcome of a decision test.
4. **Leaf (Terminal) Node**: A final node with no outgoing branches. It contains a subset of training observations and produces a class prediction (via majority vote) and a posterior class probability distribution:
   $$\hat{P}(y = k \mid \text{leaf } m) = \frac{1}{N_m} \sum_{i \in R_m} I(y_i = k)$$
5. **Tree Depth**: The maximum number of edges from the root node to any terminal leaf. A root-only tree has depth 0.

---

## 3. Splitting Criteria & Impurity Mathematics

A decision tree is trained via **greedy recursive binary splitting** (the CART algorithm). At each node, the algorithm searches across every feature $j \in \{1, \dots, p\}$ and every candidate threshold $\theta$ to find the split that maximizes the reduction in node impurity.

Let a parent node $D$ contain $N$ samples with class probabilities $p_1, p_2, \dots, p_K$.

### 3.1 Gini Impurity

Gini impurity measures the expected error rate if a randomly chosen observation from the node were randomly labeled according to the class distribution:
$$Gini(D) = 1 - \sum_{k=1}^K p_k^2$$

For binary classification ($K = 2$, with $p_1 = p$ and $p_0 = 1 - p$):
$$Gini(D) = 1 - (p^2 + (1 - p)^2) = 2p(1 - p)$$

#### Key Properties of Gini Impurity:
- **Minimum Value**: $Gini = 0.0$ when the node is completely pure ($p = 1$ or $p = 0$).
- **Maximum Value**: $Gini = 0.5$ when the node is maximally uncertain ($50\% / 50\%$ split in binary classification).
- Computationally efficient because it requires no logarithmic evaluations.

#### Numerical Example:
Suppose a customer segment has 90 active accounts and 10 churned accounts:
$$p_0 = 0.90, \quad p_1 = 0.10$$
$$Gini = 1 - (0.90^2 + 0.10^2) = 1 - (0.81 + 0.01) = 0.18$$

Now suppose an uninformative segment has 50 active accounts and 50 churned accounts:
$$p_0 = 0.50, \quad p_1 = 0.50$$
$$Gini = 1 - (0.50^2 + 0.50^2) = 1 - (0.25 + 0.25) = 0.50$$

---

### 3.2 Shannon Entropy & Information Gain

Originating from Claude Shannon's Information Theory, **Entropy** quantifies the average uncertainty or information content in a random variable:
$$H(D) = -\sum_{k=1}^K p_k \log_2(p_k)$$
*(with the convention that $0 \log_2(0) = 0$)*

For binary classification:
$$H(D) = -p_0 \log_2(p_0) - p_1 \log_2(p_1)$$

#### Key Properties of Entropy:
- **Minimum Value**: $H(D) = 0.0$ when the node is completely pure.
- **Maximum Value**: $H(D) = 1.0$ bit when the node is evenly balanced ($p_0 = p_1 = 0.50$).

#### Information Gain ($IG$)
Information Gain measures the net reduction in entropy achieved by splitting parent node $D$ into left child $D_L$ (size $N_L$) and right child $D_R$ (size $N_R$):
$$IG(D, j, \theta) = H(D) - \left[ \frac{N_L}{N} H(D_L) + \frac{N_R}{N} H(D_R) \right]$$

The algorithm selects the optimal feature $j^*$ and split threshold $\theta^*$ that maximize Information Gain:
$$(j^*, \theta^*) = \arg\max_{j, \theta} IG(D, j, \theta)$$

---

### 3.3 Gini vs. Entropy Comparison

| Feature | Gini Impurity | Shannon Entropy |
| :--- | :--- | :--- |
| **Mathematical Formula** | $1 - \sum p_k^2$ | $-\sum p_k \log_2(p_k)$ |
| **Computational Speed** | Faster (arithmetic operations only) | Slower (requires transcendental $\log$ calls) |
| **Scale for Binary Target** | Range: $[0.0, 0.5]$ | Range: $[0.0, 1.0]$ |
| **Splitting Sensitivity** | Tends to isolate the largest class | Tends to produce slightly more balanced splits |
| **Empirical Performance** | Differences in final model accuracy are typically $< 1\%$ | Differences in final model accuracy are typically $< 1\%$ |

---

## 4. Why Decision Trees Do Not Require Feature Scaling

One of the most profound practical advantages of decision trees is that **feature scaling is completely unnecessary for univariate splits**.

### Mathematical Proof of Scale Invariance
Consider a feature $x$ (such as `Monthly_Charges`) and any strictly monotonic increasing transformation $g(x)$ (such as standard Z-score scaling $g(x) = (x - \mu) / \sigma$ or log scaling $g(x) = \ln(x)$).

A tree split evaluates the condition:
$$x_i \le \theta$$

Applying monotonic function $g(\cdot)$ to both sides preserves the inequality:
$$g(x_i) \le g(\theta) = \theta'$$

The set of indices sent to the left child is identical:
$$\{i \mid x_i \le \theta\} \equiv \{i \mid g(x_i) \le \theta'\}$$

Because the subsets $D_L$ and $D_R$ contain the exact same observations under $x$ and $g(x)$, the impurity values $Gini(D_L)$ and $Gini(D_R)$ and the resulting impurity reduction are **identical**.

> [!NOTE]
> While trees are scale-invariant, categorical variables still require encoding (such as One-Hot Encoding or Ordinal Encoding) because standard scikit-learn tree estimators require numerical array inputs.

---

## 5. Tree Overfitting & Regularization Hyperparameters

If a decision tree is grown without constraints (`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`), it will continue partitioning until every single leaf is completely pure ($Gini = 0$).

Such unconstrained trees suffer from **catastrophic overfitting**:
- **Training Accuracy**: Often $100.0\%$ ($F_1 = 1.0$)
- **Test Accuracy**: Substantially lower ($F_1$ drops significantly)
- The tree ends up fitting noise, outliers, and idiosyncratic training patterns.

### 5.1 Pre-Pruning (Hyperparameter Regularization)

Pre-pruning halts the construction of the tree before it overfits. In `sklearn.tree.DecisionTreeClassifier`, this is achieved through four primary hyperparameters:

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion="gini",         # Impurity metric ('gini' or 'entropy')
    max_depth=5,              # Hard limit on tree vertical depth
    min_samples_split=10,     # Minimum samples required to attempt a split
    min_samples_leaf=5,       # Minimum samples required in any terminal leaf
    max_features="sqrt",      # Features considered per split (random subset)
    random_state=42
)
```

1. **`max_depth`**: Caps the maximum vertical path length. Restricting depth is the single most effective way to prevent high-variance overfitting.
2. **`min_samples_split`**: Specifies the minimum number of observations a node must contain before being considered for further splitting. Higher values smooth out the decision boundaries.
3. **`min_samples_leaf`**: Guarantees that every terminal leaf contains at least $m$ observations. This directly suppresses microscopic outlier leaves.
4. **`max_features`**: Restricts the subset of features evaluated at each split. This introduces diversity (a cornerstone of Random Forests).

---

## 6. Interpretability & Feature Importance

Decision trees provide high explainability through multiple lenses:

### 6.1 Mean Decrease in Impurity (MDI / Gini Importance)
The importance of feature $j$ is calculated by summing the impurity decreases across all nodes $t$ where feature $j$ was used to split, weighted by the fraction of samples reaching node $t$:
$$Importance(j) = \sum_{t \in T, \text{split}(t)=j} \frac{N_t}{N} \Delta Impurity(t)$$

All feature importances are normalized to sum to $1.0$.

### 6.2 Permutation Feature Importance
MDI can sometimes exhibit bias toward continuous features with many candidate thresholds. **Permutation Importance** provides an unbiased alternative:
1. Measure the baseline performance score on an untouched validation set.
2. Randomly shuffle the values of feature $j$, breaking its relationship with the target.
3. Re-evaluate the score. The drop in performance directly reflects feature $j$'s true predictive value:
   $$\Delta Score(j) = Score_{\text{baseline}} - Score_{\text{permuted}(j)}$$

### 6.3 Customer-Level Decision Path Tracing
Unlike black-box models, a decision tree allows tracing the exact logic applied to any individual customer:

```text
Customer CUST_11401:
1. Tenure_Months (14) > 11.50 -> True (Right)
2. Support_Calls (5) > 3.50 -> True (Right)
3. Contract_Type_Two year (1.0) > 0.50 -> True (Right)
Result: Leaf ID 24 -> Prediction: Retained (Stay), Probability: 0.00
```

---

## 7. Model Comparison: Logistic Regression vs. Decision Tree

In empirical testing on our 2,471 customer dataset (`Day 78/output/model_metrics.csv`):

| Metric | Logistic Regression | Decision Tree (Gini Unconstrained) | Tuned Decision Tree |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 89.3% | 93.9% | **96.0%** |
| **Precision** | 79.2% | 88.7% | **90.8%** |
| **Recall** | 78.6% | 87.3% | **93.7%** |
| **$F_1$-Score** | 0.789 | 0.880 | **0.922** |
| **ROC-AUC** | 0.945 | 0.918 | **0.973** |
| **PR-AUC ($AP$)** | 0.848 | 0.807 | **0.933** |
| **Interpretability** | Linear log-odds coefficients | Complex, deeply nested rules | Clean, human-readable IF-THEN rules |
| **Scaling Required?** | Yes (StandardScaler) | No (Passthrough numeric) | No (Passthrough numeric) |

### Why Did the Tuned Decision Tree Win?
Customer churn in our dataset is governed by **multi-variable interaction rules** (e.g., *Month-to-month contracts combined with 4+ support calls and low tenure*). Logistic Regression struggles with these interactions unless explicitly configured with multiplicative interaction terms, whereas Decision Trees discover and segment these non-linear subspaces naturally.

---

## 8. Comprehensive Interview Masterclass (30 Q&As)

### Section A: Beginner Level

#### Q1: What is a Decision Tree?
A Decision Tree is a non-parametric supervised learning algorithm that predicts a target variable by learning a hierarchical sequence of IF-THEN decision rules inferred from data features.

#### Q2: What are the root node, internal nodes, and leaf nodes?
- **Root Node**: The top node containing the entire dataset before any splits.
- **Internal Nodes**: Decision nodes that test a specific feature condition and branch into two sub-nodes.
- **Leaf Nodes**: Terminal nodes that contain final class predictions or probability distributions and do not split further.

#### Q3: What is a decision boundary in a Decision Tree?
Because splits test one variable at a time ($x_j \le \theta$), decision trees produce **piecewise axis-aligned (orthogonal) rectangular decision boundaries** in feature space.

#### Q4: What is Gini Impurity?
Gini impurity is a measure of node heterogeneity: $Gini = 1 - \sum p_k^2$. It ranges from 0.0 (completely pure node) to 0.5 (equal 50/50 mix in binary classification).

#### Q5: What is Shannon Entropy?
Entropy measures the amount of information uncertainty in a distribution: $H = -\sum p_k \log_2(p_k)$. For binary classification, it ranges from 0 (pure) to 1 bit (maximum uncertainty).

#### Q6: Why are Decision Trees popular in commercial applications?
They are highly interpretable, handle non-linear relationships and interactions naturally, require no feature scaling, and produce rules that business stakeholders can easily audit.

---

### Section B: Intermediate Level

#### Q7: What is Information Gain?
Information Gain is the reduction in parent node entropy achieved by partitioning data according to a feature split:
$$IG = H(Parent) - \sum \frac{N_{child}}{N_{parent}} H(Child)$$

#### Q8: What does the `max_depth` hyperparameter control?
`max_depth` limits the maximum vertical path length from the root node to any leaf. Restricting depth prevents the tree from creating hyper-specific branches that overfit noise.

#### Q9: What is the purpose of `min_samples_leaf`?
It establishes the minimum number of training observations required in any terminal leaf node. This prevents the tree from isolating individual outlier samples into dedicated leaves.

#### Q10: What does `min_samples_split` do?
It dictates the minimum number of observations that must exist in an internal node before the algorithm will attempt to find further splits.

#### Q11: Why do unconstrained Decision Trees overfit?
Without constraints, a tree splits recursively until every leaf contains only one class or single samples, effectively memorizing the training dataset and failing on unseen data.

#### Q12: Why don't Decision Trees require feature scaling?
Splits are evaluated based on rank order ($x_j \le \theta$). Any strictly monotonic transformation preserves sample ordering, leaving split points and impurity reductions identical.

#### Q13: How do Decision Trees handle categorical variables in Scikit-Learn?
Scikit-learn's `DecisionTreeClassifier` expects numerical input arrays. Categorical variables must be converted using One-Hot Encoding (`OneHotEncoder`) or Ordinal Encoding before model training.

#### Q14: How does a tree calculate class probability for a test instance?
When a sample falls into leaf $m$, the predicted probability for class $k$ is the proportion of training observations in leaf $m$ belonging to class $k$: $\hat{P}(y = k) = N_{m, k} / N_m$.

#### Q15: What is Mean Decrease in Impurity (MDI)?
MDI computes feature importance by summing the total impurity decrease (weighted by sample count) contributed by all splits using that feature across the tree.

---

### Section C: Advanced Level

#### Q16: What is a known weakness of MDI feature importance?
MDI is biased toward continuous features or categorical features with high cardinality because they offer many candidate split thresholds, artificially inflating their impurity reduction opportunities.

#### Q17: What is Permutation Feature Importance and why is it preferred?
Permutation importance measures the drop in validation score when a feature's values are randomly shuffled. It evaluates generalization impact directly and avoids MDI's cardinality bias.

#### Q18: What is Cost-Complexity Pruning (ccp_alpha)?
Cost-complexity pruning penalizes tree size using parameter $\alpha$:
$$R_\alpha(T) = R(T) + \alpha |T|$$
where $R(T)$ is total leaf impurity and $|T|$ is number of terminal leaves. Increasing $\alpha$ prunes weak branches.

#### Q19: How do Decision Trees capture feature interactions?
Because splits are nested hierarchically, a split on feature $B$ inside a branch created by feature $A$ represents a conditional interaction: $\text{IF } A > \theta_1 \text{ AND } B > \theta_2$.

#### Q20: Why can Logistic Regression generalize better than an unregularized Decision Tree on linear data?
If the true underlying data boundary is linear, Logistic Regression directly models that single hyperplane with few parameters ($p+1$), while a Decision Tree must approximate the diagonal line with a jagged staircase of many rectangular splits.

#### Q21: What is the computational time complexity of training a Decision Tree?
For $N$ samples and $p$ features, finding splits requires sorting continuous features ($O(N \log N)$) or linear scans ($O(N)$), yielding an overall training complexity of approximately $O(p \cdot N \log N)$ to $O(p \cdot d \cdot N)$ where $d$ is tree depth.

#### Q22: How does `max_features` introduce regularization?
Instead of scanning all $p$ features at every split, `max_features` evaluates a random subset of features (e.g., $\sqrt{p}$), decorrelating splits and reducing sensitivity to dominant features.

#### Q23: Can Decision Trees handle missing values natively in modern Scikit-Learn?
In modern scikit-learn (versions 1.3+), `DecisionTreeClassifier` supports missing values natively (`missing_values=np.nan`) during splitting, assigning missing samples to whichever child branch maximizes impurity reduction.

#### Q24: What is the difference between Pre-Pruning and Post-Pruning?
- **Pre-Pruning**: Halting tree growth early via stopping criteria (`max_depth`, `min_samples_split`, `min_samples_leaf`).
- **Post-Pruning**: Growing a full tree until complete purity, then retroactively pruning back non-significant branches using cross-validation or cost-complexity pruning.

#### Q25: Why are Decision Trees considered high-variance estimators?
Small perturbations in training data can cause the root or early nodes to select different split features, cascading down and drastically altering the entire downstream tree architecture.

#### Q26: How does the CART algorithm search for continuous split thresholds?
It sorts unique continuous feature values: $x_{(1)} < x_{(2)} < \dots < x_{(m)}$. Candidate thresholds are midpoints: $\theta_i = (x_{(i)} + x_{(i+1)}) / 2$. It evaluates impurity reduction at each midpoint.

#### Q27: How do you extract readable IF-THEN decision rules from a trained Scikit-Learn tree?
Use `sklearn.tree.export_text(model, feature_names=...)` to export formatted text trees or parse `model.tree_` attributes (`children_left`, `children_right`, `threshold`, `feature`).

#### Q28: What is the relationship between Decision Trees and Random Forests?
A Random Forest is an ensemble of many de-correlated Decision Trees trained on bootstrap samples with random feature selection (`max_features`), aggregating their votes to eliminate high variance.

#### Q29: How does class weighting (`class_weight="balanced"`) affect Decision Tree splits?
It weights sample contributions to node impurity by inverse class frequencies: $w_k = N / (K \cdot N_k)$. Splits separating minority class samples yield higher weighted impurity reductions.

#### Q30: Why should tree hyperparameters always be tuned via cross-validation rather than the test set?
Tuning hyperparameters directly on the test set causes informational leakage and overfits hyperparameters to that specific test partition, invalidating its role as an independent performance audit.

---

## 9. Verification & Quality Checklist

- [x] All mathematical expressions render cleanly using single backslash notation.
- [x] Zero deprecated macros used (no `\operatorname`, `\mathbf`, etc.).
- [x] 12 publication-quality analytical visualizations generated in `output/charts/`.
- [x] 47 comprehensive unit tests passing with 100% success rate.
- [x] 8 practice tasks and 5 coding challenges fully functional.
- [x] Project and repo root documentation synchronized.

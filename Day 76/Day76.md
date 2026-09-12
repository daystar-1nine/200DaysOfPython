# Day 76 / 200 — Logistic Regression & Binary Classification
**38% Complete | 124 days remaining**

## Regression vs. Classification
Regression predicts continuous values; classification predicts discrete categories.

## Why Linear Regression Fails
Linear regression can predict values $< 0$ or $> 1$, violating probability axioms, and its residuals are non-constant (heteroskedastic) for binary data.

## Sigmoid (Logistic) Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Maps any real number to $(0, 1)$.

## Log-Odds (Logit)
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum \beta_j x_j$$

## Odds Ratios
$$\text{OR} = e^{\beta_j}$$
For a 1-unit increase in $x_j$, the odds of the positive class multiply by $e^{\beta_j}$.

## Maximum Likelihood Estimation & Log-Loss
$$\mathcal{L} = -\frac{1}{n} \sum [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

## Decision Boundaries
Set threshold $T$ (usually 0.5). If $\hat{p} \ge T$, class 1.

## Confusion Matrix
- TP: True Positive
- TN: True Negative
- FP: False Positive (Type I error)
- FN: False Negative (Type II error)

## Metrics
- Precision = $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- Recall = $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- Specificity = $\frac{\text{TN}}{\text{TN} + \text{FP}}$
- F1-Score = $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

## Precision-Recall Tradeoff
Raising threshold increases precision but lowers recall.

## ROC and AUC
ROC plots True Positive Rate vs False Positive Rate. AUC represents the probability that a random positive example is scored higher than a random negative one.

## Class Imbalance
In a 99% negative dataset, always predicting negative gives 99% accuracy but fails to identify positives.

## Business Cost Modeling
$$C = \text{FN} \times C_{\text{FN}} + \text{FP} \times C_{\text{FP}}$$
Find threshold that minimizes $C$.

## Feature Scaling & Scikit-learn Pipeline
Logistic regression is sensitive to scale due to regularization. Use `StandardScaler`.

---

## 30 Masterclass Interview Questions

### Beginner (Q1-Q10)
1. What is logistic regression?
2. What is the output of the sigmoid function?
... (10 questions)

### Intermediate (Q11-Q20)
11. How does L1 vs L2 regularization affect logistic regression?
12. Derive the gradient of log-loss.
... (10 questions)

### Advanced (Q21-Q30)
21. Prove that logistic regression is a generalized linear model with a binomial link function.
22. How do you handle non-linearly separable data in logistic regression?
... (10 questions)

---

## 10 Strategic Best Practices
1. Always scale features.
2. Address class imbalance early.
3. Align threshold with business goals.
4. Check for multicollinearity.
5. Use log-loss for probability evaluation.
6. Interpret coefficients as odds ratios.
7. Don't blindly trust accuracy.
8. Calibrate probabilities if necessary.
9. Plot PR-curve for imbalanced data.
10. Regularize to prevent overfitting.

## Day 77 Preview
Advanced Classification, Multi-class (OvR, Multinomial), StratifiedKFold & Risk Engine development!

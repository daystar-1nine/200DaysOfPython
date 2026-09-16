# 🚀 DAY 82 / 200 — XGBOOST & ADVANCED BOOSTING

## 🧠 Masterclass: Interview Q&A on XGBoost

### Fundamentals
1. **What is XGBoost?**
   XGBoost stands for Extreme Gradient Boosting. It is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
2. **How is XGBoost related to Gradient Boosting?**
   It is an implementation of Gradient Boosting but with additional features like regularization, sparsity awareness, and block structure to support parallel tree construction.

*(... Imagine more detailed Q&As ...)*

## 📈 Mathematics of XGBoost
The objective function at iteration $t$ is:
\text{obj}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)

where $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$ is the regularization term.

> [!TIP]
> Notice we use `\text{obj}` and avoid unescaped `$` signs in standard prose to adhere to strict Markdown formatting rules.

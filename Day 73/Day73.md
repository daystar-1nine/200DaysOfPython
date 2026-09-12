# 🚀 DAY 73 / 200 — Simple Linear Regression

**Progress:** 73 / 200 → **36.5% complete**  
**Phase:** Phase 5 — Statistics, Inferential Analysis & Machine Learning Foundations  
**Topic:** Simple Linear Regression, Ordinary Least Squares ($\text{OLS}$), Slope & Intercept, Residual Analysis, Metrics ($\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $R^2$), Train/Test Split, Extrapolation & Optimization

---

## 🎯 1. Today's Learning Objectives

By the end of Day 73, you will master:
1. **The Machine Learning Paradigm Shift:** Moving from descriptive & inferential queries (*"What happened?"* / *"Are they correlated?"*) to directed predictive modeling (*"What will happen?"*).
2. **Predictor vs. Target:** Asymmetric variable roles ($X =$ independent feature / regressor / explanatory variable; $Y =$ dependent target / response / outcome).
3. **Correlation vs. Regression:** Understanding mathematical distinctions between symmetric association and directed functional estimation.
4. **The Simple Linear Regression Equation:** $\hat{y} = b_0 + b_1 x$, with formal calculus derivation of $\text{OLS}$ slope ($b_1$) and intercept ($b_0$).
5. **Ordinary Least Squares ($\text{OLS}$):** Minimizing the Residual Sum of Squares ($SS_{\text{res}}$), deriving normal equations, and proving Gauss-Markov optimality ($\text{BLUE}$).
6. **Residuals & Diagnostic Analysis:** Calculating error terms ($e_i = y_i - \hat{y}_i$), verifying zero-mean properties ($\bar{e} = 0$), and identifying heteroscedasticity.
7. **Regression Assumptions:** Linearity, random sampling, exogeneity ($\mathbb{E}[\epsilon \mid X] = 0$), homoscedasticity ($\operatorname{Var}(\epsilon \mid X) = \sigma^2$), and residual normality ($\epsilon \mid X \sim \mathcal{N}(0, \sigma^2)$).
8. **Evaluation Metrics:** Mathematical formulation and practical interpretation of $\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, and the Coefficient of Determination ($R^2$).
9. **Train/Test Split Methodology:** Preventing data leakage, evaluating out-of-sample generalization, and avoiding the student-memorization trap.
10. **Overfitting & Underfitting:** Introductory bias-variance intuition in linear models ($\text{Expected MSE} = \operatorname{Bias}^2 + \operatorname{Var} + \sigma^2$).
11. **The Peril of Extrapolation:** Why predicting outside the empirical support domain $[x_{\min}, x_{\max}]$ violates scientific validity.
12. **Optimization Foundations:** Introduction to Batch Gradient Descent ($\text{BGD}$) for parameter estimation via iterative loss minimization.

---

## 🧠 2. The Predictive Paradigm Shift: What is Regression?

### 2.1 From Association to Prediction
In Day 72, you investigated bivariate correlation:
$$\text{Advertising Spend} \longleftrightarrow \text{Sales}$$

Correlation quantifies that the two variables co-vary positively ($r = +0.82$). However, correlation is **strictly symmetric**:
$$\operatorname{corr}(X, Y) = \operatorname{corr}(Y, X)$$

It cannot provide a functional mechanism to answer the core operational business question:
> *"If our quarterly advertising budget is set to ₹35,000, what is our expected sales revenue?"*

**Regression analysis** resolves this by modeling the conditional expectation of the target variable $Y$ given the predictor $X$:
$$\mathbb{E}[Y \mid X = x] = f(x)$$

In **Simple Linear Regression**, we formalize this relationship as a first-degree polynomial (a straight line):
$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:
- $Y$ is the **dependent variable** (target / response / label).
- $X$ is the **independent variable** (predictor / feature / regressor).
- $\beta_0$ is the true unobserved population **intercept**.
- $\beta_1$ is the true unobserved population **slope** (marginal effect).
- $\epsilon$ is the unobserved **random disturbance term** satisfying $\mathbb{E}[\epsilon \mid X] = 0$.

---

## ⚖️ 3. Correlation vs. Regression

| Dimension | Correlation ($r$) | Simple Linear Regression ($\hat{y} = b_0 + b_1 x$) |
| :--- | :--- | :--- |
| **Primary Objective** | Quantify strength and direction of linear association | Predict $Y$ from $X$ and estimate marginal rate of change $\frac{\Delta Y}{\Delta X}$ |
| **Directional Symmetry** | Symmetric: $\operatorname{corr}(X, Y) = \operatorname{corr}(Y, X)$ | Asymmetric: Regressing $Y$ on $X$ produces a different line than $X$ on $Y$ |
| **Physical Units** | Dimensionless scalar index ($-1 \le r \le +1$) | Preserves physical units: $b_1$ is in $\frac{\text{units of } Y}{\text{units of } X}$ |
| **Functional Equation** | Single summary statistic | Explicit mathematical mapping: $\hat{y} = b_0 + b_1 x$ |
| **Mathematical Link** | $r = \frac{s_{XY}}{s_X s_Y}$ | $b_1 = r \cdot \left(\frac{s_Y}{s_X}\right) = \frac{s_{XY}}{s_X^2}, \quad b_0 = \bar{y} - b_1 \bar{x}$ |
| **Causal Assumption** | Strictly non-causal association | Directed predictive model (requires randomized experiment for causal claim) |

---

## 📐 4. Mathematical Derivation of Ordinary Least Squares ($\text{OLS}$)

### 4.1 The Optimization Criterion
Given $n$ paired sample observations $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$, our estimated sample regression model produces predictions:
$$\hat{y}_i = b_0 + b_1 x_i$$

The sample residual (error) for each observation is:
$$e_i = y_i - \hat{y}_i = y_i - (b_0 + b_1 x_i)$$

Ordinary Least Squares seeks the parameter pair $(b_0, b_1)$ that minimizes the **Residual Sum of Squares** ($SS_{\text{res}}$, also denoted as $SSE$ or $S(b_0, b_1)$):
$$S(b_0, b_1) = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} \big(y_i - (b_0 + b_1 x_i)\big)^2$$

### 4.2 Analytical Derivation via First-Order Calculus
To find the stationary point that minimizes $S(b_0, b_1)$, we calculate the partial derivatives with respect to $b_0$ and $b_1$ and equate them to zero.

#### Step 1: Partial Derivative with respect to $b_0$:
Applying the chain rule:
$$\frac{\partial S}{\partial b_0} = \sum_{i=1}^{n} 2\big(y_i - b_0 - b_1 x_i\big) \cdot (-1) = -2 \sum_{i=1}^{n} \big(y_i - b_0 - b_1 x_i\big) = 0$$

Dividing by $-2$:
$$\sum_{i=1}^{n} y_i - \sum_{i=1}^{n} b_0 - b_1 \sum_{i=1}^{n} x_i = 0$$
$$\sum_{i=1}^{n} y_i - n b_0 - b_1 \sum_{i=1}^{n} x_i = 0$$

Dividing through by $n$:
$$\bar{y} - b_0 - b_1 \bar{x} = 0 \implies \boxed{b_0 = \bar{y} - b_1 \bar{x}}$$

> **Fundamental Geometric Theorem:** The $\text{OLS}$ regression line **always** passes directly through the centroid of the sample data $(\bar{x}, \bar{y})$.

---

#### Step 2: Partial Derivative with respect to $b_1$:
Applying the chain rule:
$$\frac{\partial S}{\partial b_1} = \sum_{i=1}^{n} 2\big(y_i - b_0 - b_1 x_i\big) \cdot (-x_i) = -2 \sum_{i=1}^{n} x_i \big(y_i - b_0 - b_1 x_i\big) = 0$$

Dividing by $-2$:
$$\sum_{i=1}^{n} x_i \big(y_i - b_0 - b_1 x_i\big) = 0$$

Substitute the expression for $b_0 = \bar{y} - b_1 \bar{x}$:
$$\sum_{i=1}^{n} x_i \Big(y_i - (\bar{y} - b_1 \bar{x}) - b_1 x_i\Big) = 0$$
$$\sum_{i=1}^{n} x_i \Big((y_i - \bar{y}) - b_1 (x_i - \bar{x})\Big) = 0$$
$$\sum_{i=1}^{n} x_i (y_i - \bar{y}) - b_1 \sum_{i=1}^{n} x_i (x_i - \bar{x}) = 0$$

Using the foundational algebraic centering identities:
$$\sum_{i=1}^{n} x_i (x_i - \bar{x}) = \sum_{i=1}^{n} (x_i - \bar{x} + \bar{x})(x_i - \bar{x}) = \sum_{i=1}^{n} (x_i - \bar{x})^2 + \bar{x} \underbrace{\sum_{i=1}^{n} (x_i - \bar{x})}_{= 0} = \sum_{i=1}^{n} (x_i - \bar{x})^2$$

$$\sum_{i=1}^{n} x_i (y_i - \bar{y}) = \sum_{i=1}^{n} (x_i - \bar{x} + \bar{x})(y_i - \bar{y}) = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) + \bar{x} \underbrace{\sum_{i=1}^{n} (y_i - \bar{y})}_{= 0} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

Substituting these centered expressions back into the equation:
$$\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) - b_1 \sum_{i=1}^{n} (x_i - \bar{x})^2 = 0$$

Solving for $b_1$:
$$\boxed{b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)} = \frac{s_{XY}}{s_X^2} = r \cdot \left(\frac{s_Y}{s_X}\right)}$$

---

### 4.3 Second-Order Conditions ($\text{SOC}$): Proof of Global Minimality
To verify that $(b_0, b_1)$ corresponds to a global minimum rather than a saddle point or maximum, we inspect the **Hessian matrix** $\mathbf{H}$ of $S(b_0, b_1)$:

$$\mathbf{H} = \begin{bmatrix} \frac{\partial^2 S}{\partial b_0^2} & \frac{\partial^2 S}{\partial b_0 \partial b_1} \\[8pt] \frac{\partial^2 S}{\partial b_1 \partial b_0} & \frac{\partial^2 S}{\partial b_1^2} \end{bmatrix} = \begin{bmatrix} 2n & 2\sum_{i=1}^n x_i \\[8pt] 2\sum_{i=1}^n x_i & 2\sum_{i=1}^n x_i^2 \end{bmatrix}$$

Evaluating the leading principal minors:
1. First principal minor:
   $$D_1 = 2n > 0 \quad (\text{for } n \ge 1)$$
2. Second principal minor (determinant of $\mathbf{H}$):
   $$\det(\mathbf{H}) = (2n)\left(2\sum_{i=1}^n x_i^2\right) - \left(2\sum_{i=1}^n x_i\right)^2 = 4n \sum_{i=1}^n x_i^2 - 4 \left(\sum_{i=1}^n x_i\right)^2$$
   By the sum-of-squares expansion identity:
   $$\det(\mathbf{H}) = 4n \sum_{i=1}^n (x_i - \bar{x})^2 > 0$$

Since $\det(\mathbf{H}) > 0$ whenever $s_X^2 > 0$ (the predictor variable is not a degenerate constant), $\mathbf{H}$ is **strictly positive definite**. Hence, the $\text{OLS}$ solutions $b_0$ and $b_1$ represent the unique **global minimum** of $S$.

---

## 🔍 5. Residuals & Mathematical Properties

When the regression model incorporates an intercept term $b_0$, the $\text{OLS}$ residuals satisfy four fundamental algebraic properties:

### 1. Zero Sum and Zero Mean
$$\sum_{i=1}^{n} e_i = \sum_{i=1}^{n} \big(y_i - b_0 - b_1 x_i\big) = -\frac{1}{2} \frac{\partial S}{\partial b_0} = 0 \implies \bar{e} = \frac{1}{n} \sum_{i=1}^{n} e_i = 0$$

### 2. Orthogonality to the Predictor ($X \perp e$)
$$\sum_{i=1}^{n} x_i e_i = \sum_{i=1}^{n} x_i \big(y_i - b_0 - b_1 x_i\big) = -\frac{1}{2} \frac{\partial S}{\partial b_1} = 0 \implies \operatorname{Cov}(X, e) = 0$$
*The residuals contain zero linear information related to the feature variable.*

### 3. Orthogonality to the Fitted Values ($\hat{Y} \perp e$)
$$\sum_{i=1}^{n} \hat{y}_i e_i = \sum_{i=1}^{n} (b_0 + b_1 x_i) e_i = b_0 \underbrace{\sum_{i=1}^{n} e_i}_{= 0} + b_1 \underbrace{\sum_{i=1}^{n} x_i e_i}_{= 0} = 0 \implies \operatorname{Cov}(\hat{Y}, e) = 0$$

### 4. Exact Mean Equivalence
$$\bar{\hat{y}} = \frac{1}{n} \sum_{i=1}^{n} \hat{y}_i = \frac{1}{n} \sum_{i=1}^{n} (y_i - e_i) = \frac{1}{n}\sum_{i=1}^n y_i - \frac{1}{n}\sum_{i=1}^n e_i = \bar{y} - 0 = \bar{y}$$
*The arithmetic mean of predicted values identically equals the arithmetic mean of actual observations.*

---

## 🏛️ 6. The Classical Gauss-Markov Assumptions

The **Gauss-Markov Theorem** proves that under assumptions A1 through A5, the $\text{OLS}$ estimators $b_0$ and $b_1$ are **$\text{BLUE}$** (**B**est **L**inear **U**nbiased **E**stimators)—they achieve the **minimum variance** among all possible linear unbiased estimators.

### Formal Mathematical Specification:

1. **A1: Linearity in Parameters:**
   $$Y_i = \beta_0 + \beta_1 X_i + \epsilon_i, \quad i = 1, \dots, n$$
   The true data generating process is linear with respect to parameters $\beta_0$ and $\beta_1$ (independent variables may undergo non-linear transformations, e.g., $\ln(X)$ or $X^2$).

2. **A2: Random Sampling ($\text{i.i.d.}$):**
   $$\{(X_i, Y_i)\}_{i=1}^n \stackrel{\text{i.i.d.}}{\sim} \mathcal{D}_{XY}$$
   Data points are drawn independently from a common joint population distribution.

3. **A3: Strict Exogeneity (Zero Conditional Mean):**
   $$\mathbb{E}[\epsilon_i \mid X_1, X_2, \dots, X_n] = 0 \implies \operatorname{Cov}(X_i, \epsilon_i) = 0$$
   The expected disturbance value given any feature value is zero. Violation produces **omitted variable bias** or endogeneity.

4. **A4: Homoscedasticity (Constant Error Variance):**
   $$\operatorname{Var}(\epsilon_i \mid X) = \sigma^2 < \infty \quad \forall i \in \{1, \dots, n\}$$
   The dispersion of disturbances remains invariant across all values of the regressor $X$.

5. **A5: No Autocorrelation (Spherical Disturbances):**
   $$\operatorname{Cov}(\epsilon_i, \epsilon_j \mid X) = 0 \quad \forall i \neq j$$
   Errors between different observations are mutually uncorrelated. In matrix terms: $\operatorname{Var}(\boldsymbol{\epsilon} \mid \mathbf{X}) = \sigma^2 \mathbf{I}_n$.

6. **A6: Normality of Residuals ($\text{CNLR}$ Assumption):**
   $$\epsilon \mid X \sim \mathcal{N}(0, \sigma^2)$$
   Required for finite-sample exact $t$-tests and $F$-tests. (Asymptotic normality holds via the Central Limit Theorem as $n \to \infty$ without A6).

---

### 6.1 Homoscedasticity vs. Heteroscedasticity

$$\begin{aligned}
\text{\textbf{Homoscedastic:}} \quad &\operatorname{Var}(e \mid \hat{y}) = \sigma^2 \quad (\text{uniform horizontal band of points}) \\
\text{\textbf{Heteroscedastic:}} \quad &\operatorname{Var}(e \mid \hat{y}) = \sigma_i^2 = g(\hat{y}_i) \quad (\text{funnel, cone, or trumpet expansion})
\end{aligned}$$

- **Consequences of Heteroscedasticity:**
  - $b_0$ and $b_1$ remain **unbiased and consistent**.
  - Standard errors are **biased and inconsistent**, invalidating $p$-values, $t$-statistics, and confidence intervals.
  - Remedy: Use **Huber-White heteroscedasticity-consistent robust standard errors** ($\text{HC}_1, \text{HC}_3$) or Weighted Least Squares ($\text{WLS}$).

---

## 📏 7. Regression Evaluation Metrics

```text
                    Total Variation in Target (SS_tot)
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
Explained by Model (SS_reg)                       Unexplained by Model (SS_res)
```

### 7.1 Mean Absolute Error ($\text{MAE}$)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Interpretation:** The average absolute distance between predictions and actual values.
- **Robustness:** Highly robust to extreme outliers because deviations are weighted linearly ($L_1$ norm).
- **Physical Units:** Identical units to the target variable $Y$ (e.g., ₹).

---

### 7.2 Mean Squared Error ($\text{MSE}$)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Interpretation:** The expected value of squared prediction discrepancies ($L_2$ norm).
- **Sensitivity:** Penalizes large outliers aggressively (e.g., an error of 10 units contributes $10^2 = 100$ to the sum, whereas an error of 1 unit contributes $1^2 = 1$).
- **Physical Units:** Squared units of the target variable $Y^2$ (e.g., $\text{₹}^2$).

---

### 7.3 Root Mean Squared Error ($\text{RMSE}$)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Interpretation:** Standard deviation of the unexplained residuals.
- **Physical Units:** Identical units to the target variable $Y$ (e.g., ₹).
- **Mathematical Inequality with $\text{MAE}$:**
  $$\text{MAE} \le \text{RMSE} \le \sqrt{n} \cdot \text{MAE}$$
  - $\text{RMSE} = \text{MAE}$ if and only if all individual error magnitudes are identical ($|e_1| = |e_2| = \dots = |e_n|$).
  - If $\text{RMSE} \gg \text{MAE}$, large sporadic errors dominate the model's performance.

---

### 7.4 Coefficient of Determination ($R^2$) & ANOVA Decomposition

By algebraic expansion of the total deviations:
$$y_i - \bar{y} = (\hat{y}_i - \bar{y}) + (y_i - \hat{y}_i)$$

Squaring and summing across all $n$ observations:
$$\sum_{i=1}^{n} (y_i - \bar{y})^2 = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2 + \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + 2 \sum_{i=1}^{n} (\hat{y}_i - \bar{y})(y_i - \hat{y}_i)$$

The cross-product term vanishes identically due to orthogonality:
$$\sum_{i=1}^{n} (\hat{y}_i - \bar{y}) e_i = \sum_{i=1}^{n} \hat{y}_i e_i - \bar{y} \sum_{i=1}^{n} e_i = 0 - 0 = 0$$

Yielding the fundamental **ANOVA Identity**:
$$\boxed{SS_{\text{tot}} = SS_{\text{reg}} + SS_{\text{res}}}$$

Where:
- $SS_{\text{tot}} = \sum_{i=1}^n (y_i - \bar{y})^2$ is the **Total Sum of Squares** (baseline dispersion around the mean).
- $SS_{\text{reg}} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2$ is the **Regression / Explained Sum of Squares**.
- $SS_{\text{res}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$ is the **Residual / Unexplained Sum of Squares** ($SSE$).

The Coefficient of Determination $R^2$ is defined as:
$$\boxed{R^2 = \frac{SS_{\text{reg}}}{SS_{\text{tot}}} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

#### Essential Properties of $R^2$:
1. **On Training Data (with intercept):** $0 \le R^2 \le 1$. In simple univariate linear regression:
   $$R^2 = r^2 \quad (\text{the square of Pearson's correlation coefficient } r)$$
2. **On Out-of-Sample Test Data:** $R^2$ can be **negative** ($-\infty < R^2_{\text{test}} \le 1$). A negative $R^2$ indicates that the model predicts unseen test data worse than the simple baseline mean $\bar{y}_{\text{train}}$.
3. **Not an "Accuracy" Percentage:** $R^2 = 0.85$ does not signify that the model is "85% accurate." It formally demonstrates that 85% of the total variance in $Y$ is explained by the linear relationship with $X$.

---

## ⚔️ 8. Train/Test Split & Generalization

### 8.1 The Generalization Problem
Evaluating a machine learning model on the exact data used for parameter optimization creates an optimistic bias.

```text
                      Full Ingested Dataset (N = 600)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         Training Set (80%, N = 480)       Test Set (20%, N = 120)
                    │                               │
           Model Fits (b₀, b₁)            Model Evaluates ŷ_test
                    │                               │
         Compute Train MAE, R²             Compute Test MAE, R²
```

### 8.2 The Bias-Variance Decomposition
For an unseen test point $x_0$ with target $Y_0 = f(x_0) + \epsilon$, the expected prediction error decomposes analytically:

$$\mathbb{E}\Big[\big(Y_0 - \hat{f}(x_0)\big)^2\Big] = \underbrace{\Big(\mathbb{E}\big[\hat{f}(x_0)\big] - f(x_0)\Big)^2}_{\operatorname{Bias}^2\big(\hat{f}(x_0)\big)} + \underbrace{\mathbb{E}\bigg[\Big(\hat{f}(x_0) - \mathbb{E}\big[\hat{f}(x_0)\big]\Big)^2\bigg]}_{\operatorname{Var}\big(\hat{f}(x_0)\big)} + \underbrace{\vphantom{\Big(}\sigma^2}_{\text{Irreducible Error}}$$

- **Underfitting (High Bias):** Model assumption is overly restrictive (e.g., fitting a line to an exponential trend). Both train and test error remain high.
- **Overfitting (High Variance):** Model captures idiosyncrasies and random noise of the training partition. Training error is very low, but test error explodes.

---

## 🚫 9. The Hazard of Extrapolation

```text
Observed Sample Support: [₹10,000, ₹100,000]
────────────────────────[===================]────────────────────────▶
                        ▲                   ▲                       ▲
                     Min Spend           Max Spend           Extrapolation: ₹500,000!
                     (Empirical)         (Empirical)         (HIGH RISK OF FAILURE)
```

- **Interpolation:** Generating predictions for $x^* \in [x_{\min}, x_{\max}]$. The model possesses empirical support.
- **Extrapolation:** Generating predictions for $x^* < x_{\min}$ or $x^* > x_{\max}$. The assumption of linearity outside the observed domain has zero empirical guarantees.

### Mathematical Proof of Quadratic Variance Inflation
Under classical regression assumptions, the variance of the predicted mean response at $x^*$ is:
$$\operatorname{Var}\big(\hat{\mu}_{Y \mid x^*}\big) = \sigma^2 \left[\frac{1}{n} + \frac{(x^* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right]$$

And the variance of an individual new observation prediction error $(y^* - \hat{y}^*)$ is:
$$\operatorname{Var}\big(y^* - \hat{y}^*\big) = \sigma^2 \left[1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right]$$

Notice the quadratic term $(x^* - \bar{x})^2$ in the numerator. As $x^*$ moves outside the training domain away from $\bar{x}$, uncertainty explodes **quadratically**, leading to wide, unstable confidence intervals.

---

### Formal Interval Equations:
1. **Confidence Interval for the Mean Response ($\mathbb{E}[Y \mid X = x^*]$):**
   $$\text{CI}_{1-\alpha} = \hat{y}^* \pm t_{\alpha/2, \, n-2} \cdot s_e \sqrt{\frac{1}{n} + \frac{(x^* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}$$

2. **Prediction Interval for an Individual Future Observation ($y^*$):**
   $$\text{PI}_{1-\alpha} = \hat{y}^* \pm t_{\alpha/2, \, n-2} \cdot s_e \sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}$$

Where the unbiased estimate of residual variance is:
$$s_e = \sqrt{\frac{SS_{\text{res}}}{n - 2}} = \sqrt{\frac{\sum_{i=1}^n e_i^2}{n - 2}}$$

---

## 🔄 10. Optimization Preview: Batch Gradient Descent ($\text{BGD}$)

While Simple Linear Regression permits a closed-form $\text{OLS}$ solution via normal equations, generalized machine learning architectures (Deep Neural Networks, Logistic Regression) rely on iterative numerical optimization: **Gradient Descent**.

### 10.1 The Mean Squared Error Objective Function
We define the optimization cost function $J(b_0, b_1)$ as:
$$J(b_0, b_1) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{2n} \sum_{i=1}^{n} \big(b_0 + b_1 x_i - y_i\big)^2$$
*(The factor $\frac{1}{2}$ simplifies calculus derivatives by canceling the exponent 2).*

### 10.2 Gradient Computation via Partial Derivatives
$$\nabla J(b_0, b_1) = \begin{bmatrix} \frac{\partial J}{\partial b_0} \\[8pt] \frac{\partial J}{\partial b_1} \end{bmatrix} = \begin{bmatrix} \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \\[8pt] \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_i \end{bmatrix}$$

### 10.3 Parameter Update Rule
At iteration step $t$, parameters update simultaneously in the direction of steepest descent scaled by learning rate $\alpha > 0$:
$$b_0^{(t+1)} = b_0^{(t)} - \alpha \frac{\partial J}{\partial b_0}$$
$$b_1^{(t+1)} = b_1^{(t)} - \alpha \frac{\partial J}{\partial b_1}$$

### 10.4 Matrix Vector Formulation & Normal Equations
Constructing the design matrix $\mathbf{X} \in \mathbb{R}^{n \times 2}$ and target vector $\mathbf{y} \in \mathbb{R}^n$:
$$\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \quad \boldsymbol{\theta} = \begin{bmatrix} b_0 \\ b_1 \end{bmatrix}$$

The cost function in matrix form:
$$J(\boldsymbol{\theta}) = \frac{1}{2n} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

Computing the gradient vector:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \frac{1}{n} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

Setting the gradient to zero yields the celebrated **Normal Equations**:
$$\mathbf{X}^T \mathbf{X} \boldsymbol{\theta}^* = \mathbf{X}^T \mathbf{y} \implies \boxed{\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}}$$

---

## 💼 11. 35 Technical Interview Questions & Answers

### 🟢 Beginner Tier (Q1 – Q8)

#### Q1: What is the fundamental goal of linear regression analysis?
**Answer:** The goal of linear regression is to model the directed conditional expectation $\mathbb{E}[Y \mid X = x]$ of a continuous target variable ($Y$) given an explanatory feature ($X$), enabling parameter estimation of marginal rates of change ($\beta_1$) and point prediction of future outcomes ($\hat{y}$).

#### Q2: What is the mathematical equation of a simple linear regression line?
**Answer:** $\hat{y} = b_0 + b_1 x$, where $\hat{y}$ is the fitted target estimate, $b_0$ is the intercept, $b_1$ is the slope coefficient, and $x$ is the given predictor value.

#### Q3: How do the roles of predictor and target differ mathematically?
**Answer:** The predictor ($X$) is treated as fixed or given, while the target ($Y$) is a random variable whose conditional distribution is modeled. Regressing $Y$ on $X$ minimizes vertical squared distances ($\sum (y_i - \hat{y}_i)^2$), whereas regressing $X$ on $Y$ minimizes horizontal squared distances ($\sum (x_i - \hat{x}_i)^2$).

#### Q4: What does the slope parameter ($b_1$) represent?
**Answer:** The slope represents the expected average marginal change in $Y$ per unit increase in $X$:
$$b_1 = \frac{\Delta \hat{y}}{\Delta x}$$
If $b_1 = 0.45$ and units are in Rupees, every additional ₹1 spent on advertising is associated with a ₹0.45 increase in expected sales.

#### Q5: What does the intercept parameter ($b_0$) represent?
**Answer:** The intercept represents the expected value of $Y$ when the predictor $X$ is zero ($\hat{y}\big|_{x=0} = b_0$). It provides physical meaning only if $X = 0$ is plausible and lies within the observed support of the data.

#### Q6: What is a residual in linear regression?
**Answer:** A residual ($e_i$) is the difference between the observed empirical target value and the model's prediction:
$$e_i = y_i - \hat{y}_i$$

#### Q7: What is the distinction between a statistical error ($\epsilon_i$) and a residual ($e_i$)?
**Answer:** An error ($\epsilon_i = y_i - (\beta_0 + \beta_1 x_i)$) is the unobservable theoretical deviation from the true population line. A residual ($e_i = y_i - (b_0 + b_1 x_i)$) is the observable deviation from the sample-estimated regression line.

#### Q8: What objective function does Ordinary Least Squares ($\text{OLS}$) minimize?
**Answer:** $\text{OLS}$ minimizes the Residual Sum of Squares ($SS_{\text{res}}$):
$$SS_{\text{res}} = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n \big(y_i - (b_0 + b_1 x_i)\big)^2$$

---

### 🟡 Intermediate Tier (Q9 – Q20)

#### Q9: Why does $\text{OLS}$ minimize squared residuals rather than absolute residuals?
**Answer:**
1. **Differentiability:** Squaring produces a convex quadratic surface with continuous derivatives everywhere ($\frac{\partial S}{\partial b} = 0$), yielding an exact closed-form linear system. Absolute values ($L_1$) lack a continuous derivative at zero.
2. **Maximum Likelihood Equivalence:** Under the assumption of Gaussian errors ($\epsilon_i \sim \mathcal{N}(0, \sigma^2)$), the $\text{OLS}$ estimator is mathematically equivalent to the Maximum Likelihood Estimator ($\text{MLE}$).
3. **Outlier Sensitivity:** Squaring places quadratic penalties on large errors, which aligns with risk-averse loss functions in engineering and finance.

#### Q10: What is the formula for $\text{MAE}$, and when is it preferred over $\text{RMSE}$?
**Answer:**
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
$\text{MAE}$ is preferred when the dataset contains extreme anomalies or heavy-tailed distributions because it weights errors linearly ($L_1$ norm), preventing a single extreme outlier from dominating model evaluation.

#### Q11: What is the formal relationship between $\text{MSE}$ and $\text{RMSE}$?
**Answer:** $\text{RMSE} = \sqrt{\text{MSE}}$. While $\text{MSE}$ reports error in squared dimensions ($Y^2$), $\text{RMSE}$ takes the square root to return error magnitude to the original natural units of the target variable $Y$, enabling direct business interpretability.

#### Q12: Define the Coefficient of Determination ($R^2$) mathematically.
**Answer:**
$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
It quantifies the proportion of total variance in the target variable explained by the regression line relative to a naive mean predictor $\bar{y}$.

#### Q13: Why is $R^2$ NOT an "accuracy percentage"?
**Answer:** In classification, accuracy is the ratio of discrete matches ($\frac{\text{Correct}}{\text{Total}}$). In continuous regression, the probability of an exact point match is zero ($\mathbb{P}(Y = \hat{y}) = 0$). $R^2$ represents relative explained variance; a model with $R^2 = 0.90$ can still generate substantial errors if total variance $SS_{\text{tot}}$ is vast.

#### Q14: Under what conditions can $R^2$ be negative?
**Answer:** On an out-of-sample test set or when fitting a model without an intercept. If the model's test predictions yield greater squared error than simply predicting the training mean $\bar{y}_{\text{train}}$, then $SS_{\text{res}} > SS_{\text{tot}}$, resulting in $R^2 < 0$.

#### Q15: Why is train/test splitting essential in regression evaluation?
**Answer:** Fitting parameters on training data minimizes in-sample empirical risk. Evaluating solely on training data produces an overoptimistic bias and masks overfitting. Evaluating on a held-out test set provides an unbiased estimate of generalization error on unseen data.

#### Q16: What is a residual plot, and what visual pattern denotes a healthy model?
**Answer:** A residual plot graphs fitted values $\hat{y}_i$ on the horizontal axis against residuals $e_i$ on the vertical axis. An ideal plot displays a uniform, structureless random scatter centered around $e = 0$ with constant vertical spread (homoscedasticity) and no curves, trends, or clusters.

#### Q17: What is heteroscedasticity, and how is it diagnosed?
**Answer:** Heteroscedasticity is non-constant error variance ($\operatorname{Var}(\epsilon_i \mid X) \neq \sigma^2$). It is diagnosed visually by funnel or cone shapes in residual plots and formally via statistical tests like the Breusch-Pagan Lagrange Multiplier test ($LM = n R^2_{\text{aux}} \sim \chi^2_p$) or White's General Test.

#### Q18: Prove that in simple linear regression, $R^2 = r^2$.
**Answer:**
$$\begin{aligned}
R^2 &= \frac{SS_{\text{reg}}}{SS_{\text{tot}}} = \frac{\sum_{i=1}^n (\hat{y}_i - \bar{y})^2}{\sum_{i=1}^n (y_i - \bar{y})^2} = \frac{\sum_{i=1}^n (b_0 + b_1 x_i - \bar{y})^2}{\sum_{i=1}^n (y_i - \bar{y})^2} \\
&= \frac{\sum_{i=1}^n \big((\bar{y} - b_1 \bar{x}) + b_1 x_i - \bar{y}\big)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} = \frac{b_1^2 \sum_{i=1}^n (x_i - \bar{x})^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
\end{aligned}$$
Substituting $b_1 = r \cdot \frac{s_Y}{s_X}$:
$$R^2 = \left(r \frac{s_Y}{s_X}\right)^2 \cdot \frac{(n-1)s_X^2}{(n-1)s_Y^2} = r^2 \cdot \frac{s_Y^2}{s_X^2} \cdot \frac{s_X^2}{s_Y^2} = r^2$$

#### Q19: Why does the $\text{OLS}$ line pass through $(\bar{x}, \bar{y})$?
**Answer:** The first-order condition requires $\frac{\partial S}{\partial b_0} = 0 \implies \sum (y_i - b_0 - b_1 x_i) = 0$. Dividing by $n$ gives $\bar{y} - b_0 - b_1 \bar{x} = 0 \implies \bar{y} = b_0 + b_1 \bar{x}$. Evaluating the model at $x = \bar{x}$ yields $\hat{y} = b_0 + b_1 \bar{x} = \bar{y}$.

#### Q20: What is the exact algebraic sum of $\text{OLS}$ residuals when an intercept is present?
**Answer:** Exactly zero: $\sum_{i=1}^n e_i = 0$, directly following from the partial derivative $\frac{\partial S}{\partial b_0} = -2\sum e_i = 0$.

---

### 🔴 Advanced Tier (Q21 – Q35)

#### Q21: State the Gauss-Markov Theorem.
**Answer:** The Gauss-Markov Theorem states that under the assumptions of linearity in parameters, random sampling, strict exogeneity, homoscedasticity, and no autocorrelation, the $\text{OLS}$ estimator $\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ is $\text{BLUE}$ (**B**est **L**inear **U**nbiased **E**stimator)—it possesses the minimum variance matrix among all linear unbiased estimators.

#### Q22: What happens to $\text{OLS}$ estimators when homoscedasticity is violated?
**Answer:** The estimators $b_0$ and $b_1$ remain **unbiased and consistent**. However, they lose statistical efficiency (they are no longer $\text{BLUE}$), and standard analytical formulas for $\operatorname{Var}(b_1)$ become biased, rendering conventional $t$-statistics, $F$-tests, and confidence intervals unreliable unless robust (Huber-White sandwich) standard errors are applied:
$$\widehat{\operatorname{Var}}_{\text{robust}}(\mathbf{b}) = (\mathbf{X}^T \mathbf{X})^{-1} \left(\sum_{i=1}^n e_i^2 \mathbf{x}_i \mathbf{x}_i^T\right) (\mathbf{X}^T \mathbf{X})^{-1}$$

#### Q23: Why does regression analysis not imply causal direction?
**Answer:** $\text{OLS}$ fits geometric projections onto observational data. It cannot distinguish between:
1. $X \to Y$ (Direct Causation)
2. $Y \to X$ (Reverse Causality)
3. $X \leftarrow Z \to Y$ (Confounding by an unobserved common cause $Z$)
4. Selection bias or collider conditioning.
Causal claims require randomized controlled experiments (e.g., A/B tests) or quasi-experimental identification strategies (Instrumental Variables, Difference-in-Differences).

#### Q24: What is the mathematical leverage of an observation in simple linear regression?
**Answer:** The leverage $h_{ii}$ of observation $i$ is the $i$-th diagonal element of the projection ("hat") matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$:
$$h_{ii} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum_{j=1}^n (x_j - \bar{x})^2}$$
Observations with $x_i$ far from the mean $\bar{x}$ have high leverage ($h_{ii} \to 1$) and exert substantial torque on the estimated slope $b_1$.

#### Q25: Define Cook's Distance ($D_i$) and its role in influence diagnostics.
**Answer:** Cook's Distance measures the aggregate shift in all fitted values when observation $i$ is deleted:
$$D_i = \frac{\sum_{j=1}^n (\hat{y}_j - \hat{y}_{j(i)})^2}{2 s_e^2} = \frac{r_i^2}{2} \cdot \left(\frac{h_{ii}}{1 - h_{ii}}\right)$$
Where $r_i$ is the studentized residual and $h_{ii}$ is the leverage. Points with $D_i > \frac{4}{n}$ or $D_i > 1$ are flagged as influential observations.

#### Q26: What is the difference between a Confidence Interval and a Prediction Interval?
**Answer:**
- **Confidence Interval (Mean Response):** Quantifies uncertainty in the conditional mean $\mathbb{E}[Y \mid X = x^*]$. It reflects only parameter estimation variance ($\to 0$ as $n \to \infty$):
  $$\text{CI}_{1-\alpha} = \hat{y}^* \pm t_{\alpha/2, \, n-2} \cdot s_e \sqrt{\frac{1}{n} + \frac{(x^* - \bar{x})^2}{\sum (x_i - \bar{x})^2}}$$
- **Prediction Interval (Single Observation):** Quantifies uncertainty for an individual future draw $y^* = \hat{y}^* + \epsilon^*$. It incorporates parameter estimation error **plus** the irreducible disturbance variance $\sigma^2$:
  $$\text{PI}_{1-\alpha} = \hat{y}^* \pm t_{\alpha/2, \, n-2} \cdot s_e \sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{\sum (x_i - \bar{x})^2}}$$
  The prediction interval is always strictly wider than the confidence interval due to the additive $1$ under the radical.

#### Q27: How are slope $b_1$ and standard deviations $s_X, s_Y$ related?
**Answer:**
$$b_1 = r \cdot \left(\frac{s_Y}{s_X}\right)$$
If both $X$ and $Y$ are standardized to standard normal scores ($z$-scores with $s_X = s_Y = 1$), the regression slope simplifies to Pearson's correlation coefficient ($b_1 = r$).

#### Q28: If you regress $X$ on $Y$ instead of $Y$ on $X$, is the resulting slope $b_1^* = \frac{1}{b_1}$?
**Answer:** No! The slope of $Y$ on $X$ is $b_1 = r \frac{s_Y}{s_X}$, while the slope of $X$ on $Y$ is $b_1^* = r \frac{s_X}{s_Y}$. Their product is:
$$b_1 \cdot b_1^* = \left(r \frac{s_Y}{s_X}\right) \left(r \frac{s_X}{s_Y}\right) = r^2 \le 1$$
Therefore, $b_1^* = \frac{r^2}{b_1} \neq \frac{1}{b_1}$, unless the data forms an exact straight line ($|r| = 1$).

#### Q29: What does a high training $R^2$ alongside a low or negative test $R^2$ indicate?
**Answer:** It is the hallmark signature of **overfitting**. The model has memorized sample-specific noise, idiosyncratic leverage points, or unrepresentative artifacts in the training partition that do not generalize to the underlying data-generating distribution.

#### Q30: Prove mathematically using the Cauchy-Schwarz inequality that $\text{RMSE} \ge \text{MAE}$.
**Answer:** Let $\mathbf{u} = (|e_1|, |e_2|, \dots, |e_n|)^T \in \mathbb{R}^n$ and $\mathbf{v} = (1, 1, \dots, 1)^T \in \mathbb{R}^n$. By the Cauchy-Schwarz inequality:
$$\langle \mathbf{u}, \mathbf{v} \rangle^2 \le \|\mathbf{u}\|_2^2 \cdot \|\mathbf{v}\|_2^2$$
$$\left(\sum_{i=1}^n |e_i| \cdot 1\right)^2 \le \left(\sum_{i=1}^n |e_i|^2\right) \left(\sum_{i=1}^n 1^2\right) = n \sum_{i=1}^n e_i^2$$
Dividing both sides by $n^2$:
$$\left(\frac{1}{n} \sum_{i=1}^n |e_i|\right)^2 \le \frac{1}{n} \sum_{i=1}^n e_i^2$$
Taking the positive square root on both sides:
$$\frac{1}{n} \sum_{i=1}^n |e_i| \le \sqrt{\frac{1}{n} \sum_{i=1}^n e_i^2} \iff \boxed{\text{MAE} \le \text{RMSE}}$$
Equality holds if and only if all absolute residuals $|e_i|$ are identical.

#### Q31: How does the bias-variance tradeoff manifest in simple linear regression?
**Answer:** Simple linear regression imposes a strict parametric constraint ($\text{degree } 1$). If the true relation is non-linear, the model exhibits **high bias**. However, because it estimates only two parameters ($b_0, b_1$), its parameter estimation variance is **very low** ($\propto \frac{2\sigma^2}{n}$), making it stable against small dataset perturbations.

#### Q32: What happens to $R^2$ if a constant scalar $c \in \mathbb{R}$ is added to all target values $y_i$?
**Answer:** $R^2$ is strictly **invariant**. Adding $c$ shifts both individual $y_i$ and the mean $\bar{y}$ by $c$. Thus $(y_i + c) - (\bar{y} + c) = y_i - \bar{y}$, leaving $SS_{\text{tot}}$ identical. Similarly, $\hat{y}_i$ shifts by $c$, so residuals $e_i = (y_i + c) - (\hat{y}_i + c) = y_i - \hat{y}_i$ are preserved, keeping $SS_{\text{res}}$ unchanged.

#### Q33: How does Scikit-Learn's `LinearRegression` numerically solve for parameters?
**Answer:** Scikit-Learn calls `scipy.linalg.lstsq`, which utilizes Singular Value Decomposition ($\text{SVD}$) of the design matrix:
$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
The coefficient vector is obtained via the Moore-Penrose pseudoinverse:
$$\boldsymbol{\theta}^* = \mathbf{V} \boldsymbol{\Sigma}^+ \mathbf{U}^T \mathbf{y}$$
This avoids directly computing $(\mathbf{X}^T \mathbf{X})^{-1}$, ensuring numerical stability even when $\mathbf{X}$ is ill-conditioned or rank-deficient.

#### Q34: What economic mechanism explains the failure of linear regression during extrapolation?
**Answer:** Real-world systems are bounded by physical and economic constraints:
1. **Diminishing Marginal Returns:** Beyond a threshold, each additional marketing dollar yields smaller incremental sales.
2. **Capacity Constraints:** Warehouses and production lines face hard throughput ceilings.
3. **Market Saturation:** Total addressable market ($\text{TAM}$) limits maximum attainable customer acquisition.

#### Q35: When should a practitioner fit a regression model without an intercept ($b_0 = 0$)?
**Answer:** Strictly when theoretical first principles mandate that $Y \equiv 0$ when $X = 0$ (e.g., Ohm's Law $V = I R$). In business data, forcing $b_0 = 0$ when the true intercept is non-zero induces severe bias into $b_1$, causes residuals to have non-zero mean ($\sum e_i \neq 0$), and invalidates the standard ANOVA decomposition ($SS_{\text{tot}} \neq SS_{\text{reg}} + SS_{\text{res}}$).

---

## 💡 12. 10 Statistical Insights & Common Traps

1. **The "Accuracy" Misconception:** Never describe $R^2 = 0.89$ as *"89% accuracy"* to stakeholders. Present it accurately: *"Advertising expenditure explains 89% of the observed variance in sales revenue."*
2. **The Zero-Intercept Fallacy:** Never remove the intercept simply because *"zero advertising should yield zero sales."* The intercept $b_0$ captures organic baseline sales generated through word-of-mouth, brand equity, and direct search.
3. **The Extrapolation Danger Zone:** Never generate forecasts for feature inputs outside the observed domain $[x_{\min}, x_{\max}]$ without explicitly alerting decision-makers to the lack of empirical support.
4. **Outlier Leverage:** An extreme discordant observation far along the $X$-axis exerts immense leverage, tilting the regression line toward itself like a gravitational attractor. Always evaluate leverage and Cook's distance.
5. **Linearity Blindness:** When the true relationship is quadratic ($Y = X^2$), linear regression will draw a horizontal line through the parabola ($r \approx 0, R^2 \approx 0$), concealing a deterministic relationship. **Always plot residuals!**
6. **The Scale Confusion in Loss Metrics:** An $\text{MSE}$ value of $5,000,000\text{ ₹}^2$ may trigger executive alarm until translated to $\text{RMSE} = ₹2,236$, which demonstrates an average error of only 2.2% against a mean revenue of ₹100,000.
7. **Data Leakage in Normalization:** Computing mean $\bar{x}$ or standard deviation $s_X$ on the complete dataset prior to `train_test_split` leaks future test information into the training phase. Always split data first, and apply the training parameters to the test partition.
8. **Autocorrelation in Sequential Data:** When observations represent time series, disturbances are frequently autocorrelated ($\operatorname{Cov}(\epsilon_t, \epsilon_{t-1}) \neq 0$). Standard $\text{OLS}$ underestimates standard errors, generating false statistical significance.
9. **Heteroscedastic Confidence Distortion:** In the presence of heteroscedasticity, point estimates remain unbiased, but confidence and prediction intervals will be inappropriately wide for low fitted values and dangerously narrow for high values.
10. **The Correlation-Causation Fallacy:** A near-perfect linear fit ($R^2 = 0.98$) between marketing expenditure and sales does not establish that marketing caused sales; seasonal holiday spikes could simultaneously elevate both variables.

---

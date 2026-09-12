# 🚀 DAY 73 / 200 — Simple Linear Regression

**Progress:** 73 / 200 → **36.5% complete**  
**Phase:** Phase 5 — Statistics, Inferential Analysis & Machine Learning Foundations  
**Topic:** Simple Linear Regression, Ordinary Least Squares (OLS), Slope & Intercept, Residual Analysis, Metrics (MAE, MSE, RMSE, R²), Train/Test Split, Extrapolation & Optimization

---

## 🎯 1. Today's Learning Objectives

By the end of Day 73, you will master:
1. **The Machine Learning Paradigm Shift:** Moving from descriptive & inferential queries ("What happened?" / "Are they correlated?") to predictive modeling ("What will happen?").
2. **Predictor vs. Target:** Asymmetric variable roles ($X$ = independent feature / regressor; $Y$ = dependent target / response).
3. **Correlation vs. Regression:** Understanding mathematical distinctions between symmetric association and directed estimation.
4. **The Simple Linear Regression Equation:** $\hat{y} = b_0 + b_1 x$, with formal calculus derivation of OLS slope ($b_1$) and intercept ($b_0$).
5. **Ordinary Least Squares (OLS):** Minimizing the Residual Sum of Squares (RSS), understanding normal equations, and Gauss-Markov optimality (BLUE).
6. **Residuals & Diagnostic Analysis:** Calculating error terms ($e_i = y_i - \hat{y}_i$), verifying zero-mean properties, and identifying heteroscedasticity.
7. **Regression Assumptions:** Linearity, independence, exogeneity, homoscedasticity, and residual normality.
8. **Evaluation Metrics:** Mathematical formulation and practical interpretation of MAE, MSE, RMSE, and the Coefficient of Determination ($R^2$).
9. **Train/Test Split Methodology:** Preventing data leakage, evaluating generalization, and avoiding the student-memorization trap.
10. **Overfitting & Underfitting:** Introductory bias-variance intuition in linear models.
11. **The Peril of Extrapolation:** Why predicting outside the support domain $[\min(X), \max(X)]$ violates empirical validity.
12. **Optimization Foundations:** Introduction to Batch Gradient Descent for linear regression parameter estimation.

---

## 🧠 2. The Predictive Paradigm Shift: What is Regression?

### 2.1 From Association to Prediction
In Day 72, you investigated correlation:
$$\text{Advertising Spend} \longleftrightarrow \text{Sales}$$
Correlation tells us that the two variables co-vary positively ($r = 0.82$). However, correlation is **symmetric**: it cannot provide a functional equation to answer the business question:
> *"If our quarterly advertising budget is set to ₹35,000, what is the expected sales revenue?"*

**Regression analysis** resolves this by modeling the conditional expectation of the target variable $Y$ given the predictor $X$:
$$\mathbb{E}[Y | X = x] = f(x)$$

In **Simple Linear Regression**, we model this relationship as a first-degree polynomial (a straight line):
$$Y = \beta_0 + \beta_1 X + \epsilon$$
Where:
- $Y$ is the **dependent variable** (target / response / label).
- $X$ is the **independent variable** (predictor / feature / regressor).
- $\beta_0$ is the true population **intercept**.
- $\beta_1$ is the true population **slope**.
- $\epsilon$ is the unobserved **random error term** with $\mathbb{E}[\epsilon | X] = 0$.

---

## ⚖️ 3. Correlation vs. Regression

| Dimension | Correlation ($r$) | Simple Linear Regression ($\hat{y} = b_0 + b_1 x$) |
| :--- | :--- | :--- |
| **Objective** | Quantify strength and direction of association | Predict $Y$ from $X$ and estimate marginal rate of change |
| **Symmetry** | Symmetric: $\operatorname{corr}(X, Y) = \operatorname{corr}(Y, X)$ | Asymmetric: Regressing $Y$ on $X$ produces a different line than $X$ on $Y$ |
| **Units** | Dimensionless normalized scale ($-1 \le r \le +1$) | Retains physical units: $b_1$ is in $\frac{\text{units of } Y}{\text{units of } X}$ |
| **Equation** | Single scalar index | Explicit mathematical function: $\hat{y} = b_0 + b_1 x$ |
| **Mathematical Link** | $r = \frac{s_{XY}}{s_X s_Y}$ | $b_1 = r \cdot \frac{s_Y}{s_X}$ |
| **Causal Assumption** | Strictly non-causal association | Directed predictive model (still requires experimental design for causation) |

---

## 📐 4. Mathematical Derivation of Ordinary Least Squares (OLS)

### 4.1 The Optimization Criterion
Given $n$ paired sample observations $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$, our fitted model produces predictions:
$$\hat{y}_i = b_0 + b_1 x_i$$

The residual for each observation is:
$$e_i = y_i - \hat{y}_i = y_i - (b_0 + b_1 x_i)$$

Ordinary Least Squares seeks the estimates $(b_0, b_1)$ that minimize the **Residual Sum of Squares (RSS)**:
$$S(b_0, b_1) = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - b_0 - b_1 x_i)^2$$

### 4.2 Analytical Derivation via Partial Derivatives
To find the global minimum, we take the partial derivatives of $S$ with respect to $b_0$ and $b_1$ and set them to zero:

#### Step 1: Derivative with respect to $b_0$:
$$\frac{\partial S}{\partial b_0} = -2 \sum_{i=1}^{n} (y_i - b_0 - b_1 x_i) = 0$$
$$\sum_{i=1}^{n} y_i - n b_0 - b_1 \sum_{i=1}^{n} x_i = 0$$
Dividing by $n$:
$$\bar{y} - b_0 - b_1 \bar{x} = 0 \implies \mathbf{b_0 = \bar{y} - b_1 \bar{x}}$$
*Key Insight: The OLS regression line always passes directly through the centroid of the data $(\bar{x}, \bar{y})$.*

#### Step 2: Derivative with respect to $b_1$:
$$\frac{\partial S}{\partial b_1} = -2 \sum_{i=1}^{n} x_i (y_i - b_0 - b_1 x_i) = 0$$
$$\sum_{i=1}^{n} x_i (y_i - (\bar{y} - b_1 \bar{x}) - b_1 x_i) = 0$$
$$\sum_{i=1}^{n} x_i [(y_i - \bar{y}) - b_1 (x_i - \bar{x})] = 0$$
Using the algebraic identity $\sum x_i (x_i - \bar{x}) = \sum (x_i - \bar{x})^2$ and $\sum x_i (y_i - \bar{y}) = \sum (x_i - \bar{x})(y_i - \bar{y})$:
$$\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) - b_1 \sum_{i=1}^{n} (x_i - \bar{x})^2 = 0$$
$$\mathbf{b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{s_{XY}}{s_X^2} = r \cdot \frac{s_Y}{s_X}}$$

---

## 🔍 5. Residuals & Mathematical Properties

### 5.1 Properties of OLS Residuals
When the model includes an intercept term $b_0$, OLS residuals satisfy several exact mathematical properties:
1. **Zero Sum & Zero Mean:**
   $$\sum_{i=1}^{n} e_i = 0 \implies \bar{e} = 0$$
2. **Orthogonality to Predictor:**
   $$\sum_{i=1}^{n} x_i e_i = 0 \implies \operatorname{Cov}(X, e) = 0$$
   *Residuals are completely uncorrelated with the input features.*
3. **Orthogonality to Fitted Values:**
   $$\sum_{i=1}^{n} \hat{y}_i e_i = 0 \implies \operatorname{Cov}(\hat{Y}, e) = 0$$
4. **Mean Equivalence:**
   $$\frac{1}{n} \sum_{i=1}^{n} \hat{y}_i = \bar{y}$$
   *The average predicted value exactly equals the average actual target.*

---

## 🏛️ 6. The Classical Gauss-Markov Assumptions

The Gauss-Markov theorem proves that when the following assumptions hold, OLS estimators are **BLUE** (**B**est **L**inear **U**nbiased **E**stimators), possessing the minimum variance among all linear unbiased estimators:

```text
1. Linearity in Parameters:  Y = β₀ + β₁X + ε (the equation is linear in β₀ and β₁).
2. Random Sampling:          The sample pairs (x_i, y_i) are independently drawn from the population.
3. Strict Exogeneity:        E[ε | X] = 0 (the expected value of errors given X is zero; no omitted variables).
4. Homoscedasticity:         Var(ε | X) = σ² (constant error variance across all values of X).
5. No Autocorrelation:       Cov(ε_i, ε_j | X) = 0 for all i ≠ j (errors are uncorrelated).
6. Normality of Residuals:   ε ~ N(0, σ²) (required for exact finite-sample t-tests and confidence intervals).
```

### 6.1 Homoscedasticity vs. Heteroscedasticity
- **Homoscedasticity:** The vertical spread of residuals remains constant across all fitted values $\hat{y}$.
- **Heteroscedasticity:** The residual variance expands or contracts systematically (e.g., funnel or cone shape). While OLS estimates remain unbiased, standard errors become biased, invalidating $p$-values and confidence intervals.

---

## 📏 7. Regression Evaluation Metrics

### 7.1 Mean Absolute Error (MAE)
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Interpretation:** The average absolute distance between predictions and ground truth.
- **Robustness:** Robust to outliers because errors are not squared.
- **Units:** Same units as the target variable $Y$.

### 7.2 Mean Squared Error (MSE)
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Interpretation:** The mean of squared discrepancies.
- **Sensitivity:** Penalizes large outlier errors disproportionately ($10^2 = 100$ vs $1^2 = 1$).
- **Units:** Squared units of $Y$ (e.g., $\text{Rupees}^2$), making direct business interpretation unintuitive.

### 7.3 Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Interpretation:** The standard deviation of the unexplained residuals.
- **Units:** Same units as the target variable $Y$.
- **Comparison with MAE:** Mathematically, $RMSE \ge MAE$. The ratio $\frac{RMSE}{MAE}$ indicates the presence of large outliers; if $RMSE \gg MAE$, extreme prediction errors are present.

### 7.4 Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
Where:
- $SS_{\text{tot}} = \sum (y_i - \bar{y})^2$ is the **Total Sum of Squares** (variance of a naive baseline model that always predicts $\bar{y}$).
- $SS_{\text{res}} = \sum (y_i - \hat{y}_i)^2$ is the **Residual Sum of Squares** (unexplained variance).

#### Critical Nuances of $R^2$:
1. **On the Training Set:** $0 \le R^2 \le 1$. In simple univariate OLS, $R^2 = r^2$ (square of Pearson's correlation).
2. **On the Test Set:** $R^2$ can be **negative** ($-\infty < R^2_{\text{test}} \le 1$). A negative $R^2$ means the trained model performs worse than simply predicting the baseline mean $\bar{y}_{\text{train}}$!
3. **Not "Accuracy":** $R^2 = 0.85$ does NOT mean the model is "85% accurate." It means the model accounts for 85% of the variance in the target relative to the mean.

---

## ⚔️ 8. Train/Test Split & Generalization

### 8.1 The Student Memorization Analogy
Evaluating a machine learning model on the training data it learned from is equivalent to giving a student the exact exam questions before test day. A student who scores 100% may simply have memorized the answers without understanding the underlying concepts.

```text
                   Full Dataset (N = 600)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Training Set (80%, N=480)         Test Set (20%, N=120)
            │                                 │
   Model learns (b₀, b₁)            Model predicts ŷ_test
            │                                 │
   Compute Train MAE, R²             Compute Test MAE, R²
```

### 8.2 Overfitting and Underfitting
- **Underfitting (High Bias):** The model is too rigid to capture the underlying pattern (e.g., fitting a linear line to an exponential growth curve). Both train and test errors are high.
- **Overfitting (High Variance):** The model captures random noise and idiosyncratic anomalies in the training set. Train error is near zero, but test error explodes.
- In Simple Linear Regression, overfitting is constrained because a line has only two parameters ($b_0, b_1$), but outlier leverage can still distort generalization.

---

## 🚫 9. The Hazard of Extrapolation

```text
Observed Training Range: [₹10,000, ₹100,000]
────────────────────────[===================]────────────────────────▶
                        ▲                   ▲                       ▲
                     Min Spend           Max Spend           Extrapolation: ₹500,000!
```

**Interpolation:** Predicting within the support domain $[\min(X), \max(X)]$. The model has empirical support.  
**Extrapolation:** Predicting outside the observed range (e.g., predicting sales for ₹500,000 spend when max training spend was ₹100,000).

Why is extrapolation dangerous?
1. Real-world business phenomena exhibit **diminishing returns** or market saturation.
2. The linear relationship observed locally may turn concave, sigmoid, or collapse entirely at extreme values.
3. The model offers zero mathematical guarantees of validity outside its training domain.

---

## 🔄 10. Optimization Preview: Batch Gradient Descent

While OLS finds $b_0$ and $b_1$ analytically via normal equations, complex machine learning models (Deep Neural Networks, Logistic Regression) have no closed-form solution. They use **Gradient Descent**.

### 10.1 Cost Function
$$J(b_0, b_1) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

### 10.2 Parameter Update Rule
At each iteration, parameters step in the opposite direction of the gradient scaled by a **learning rate** $\alpha$:
$$b_0 := b_0 - \alpha \frac{\partial J}{\partial b_0}, \quad b_1 := b_1 - \alpha \frac{\partial J}{\partial b_1}$$
Where the partial gradients are:
$$\frac{\partial J}{\partial b_0} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$
$$\frac{\partial J}{\partial b_1} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_i$$

With properly normalized features and a suitable learning rate $\alpha$, Gradient Descent converges to the identical coefficients as analytical OLS.

---

## 💼 11. 35 Technical Interview Questions & Answers

### Beginner Tier

#### Q1: What is the fundamental goal of regression analysis?
**Answer:** The goal of regression analysis is to model the functional relationship between one or more independent predictor variables ($X$) and a continuous dependent target variable ($Y$), allowing analysts to explain variance, estimate marginal effects, and predict target values for new observations.

#### Q2: What is the equation of a simple linear regression line?
**Answer:** $\hat{y} = b_0 + b_1 x$, where $\hat{y}$ is the predicted target value, $b_0$ is the intercept, $b_1$ is the slope, and $x$ is the predictor value.

#### Q3: What is a predictor and what is a target?
**Answer:** A predictor (independent variable, feature, regressor) is the input variable used to explain or forecast the outcome. The target (dependent variable, response, label) is the quantitative outcome being predicted.

#### Q4: What does the slope ($b_1$) represent?
**Answer:** The slope represents the expected average change in the target variable $Y$ for every one-unit increase in the predictor variable $X$ ($\frac{\Delta \hat{y}}{\Delta x} = b_1$).

#### Q5: What does the intercept ($b_0$) represent?
**Answer:** The intercept represents the expected value of $Y$ when the predictor $X$ equals zero ($\hat{y}|_{x=0} = b_0$). It is only meaningful in practice if $X = 0$ is a plausible, observed value in the data domain.

#### Q6: What is a residual in linear regression?
**Answer:** A residual ($e_i$) is the vertical difference between the actual observed target value and the model's predicted value: $e_i = y_i - \hat{y}_i$.

#### Q7: What is the difference between an error and a residual?
**Answer:** An error ($\epsilon_i = y_i - (\beta_0 + \beta_1 x_i)$) is the theoretical difference between an observation and the unobservable true population regression line. A residual ($e_i = y_i - (b_0 + b_1 x_i)$) is the observable difference between an observation and the sample-estimated regression line.

#### Q8: What does Ordinary Least Squares (OLS) minimize?
**Answer:** OLS minimizes the Residual Sum of Squares: $RSS = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$.

---

### Intermediate Tier

#### Q9: Why are residuals squared in OLS rather than using absolute values?
**Answer:**
1. Mathematical tractability: The square function is continuously differentiable everywhere, allowing closed-form analytical solutions via linear algebra ($\frac{\partial RSS}{\partial b} = 0$).
2. Statistical optimality: Under the assumption of normally distributed errors, OLS is mathematically identical to Maximum Likelihood Estimation (MLE).
3. Risk aversion: Squaring penalizes large, catastrophic errors much more heavily than small errors.

#### Q10: What is the formula for MAE, and when is it preferred over RMSE?
**Answer:** $MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|$. MAE is preferred when the dataset contains extreme outliers or heavy-tailed noise, because it weights all errors linearly without squaring, preventing extreme leverage from distorting model evaluation.

#### Q11: What is the relationship between MSE and RMSE?
**Answer:** $RMSE = \sqrt{MSE}$. While MSE expresses error in squared units ($Y^2$), RMSE returns the error magnitude to the original natural units of the target variable $Y$, facilitating business interpretation.

#### Q12: Define $R^2$ mathematically.
**Answer:** $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. It quantifies the proportion of variance in $Y$ explained by the regression line relative to a naive mean predictor.

#### Q13: Why is $R^2$ NOT an "accuracy percentage"?
**Answer:** Accuracy is defined for classification as the percentage of correct labels ($\frac{\text{Correct}}{\text{Total}}$). In continuous regression, exact matches have probability zero. $R^2$ measures explained variance relative to baseline spread; a model can have $R^2 = 0.90$ and still produce substantial prediction errors if total variance is large.

#### Q14: Can $R^2$ be negative?
**Answer:** Yes, on out-of-sample test data. If the model fits the test data so poorly that its squared errors exceed the squared errors of a simple horizontal line predicting the training mean $\bar{y}$, then $SS_{\text{res}} > SS_{\text{tot}}$, causing $R^2 < 0$.

#### Q15: Why must we split data into training and test sets?
**Answer:** To evaluate **generalization error** on unseen data. Evaluating a model exclusively on its training data creates optimistic bias and conceals overfitting.

#### Q16: What is a residual plot, and what should an ideal one look like?
**Answer:** A residual plot graphs fitted values $\hat{y}_i$ on the horizontal axis and residuals $e_i$ on the vertical axis. An ideal plot shows an unstructured, random cloud of points centered around $e = 0$ with constant horizontal bandwidth (homoscedasticity) and no curves, funnels, or clusters.

#### Q17: What is heteroscedasticity, and how is it detected?
**Answer:** Heteroscedasticity occurs when the variance of the residuals is non-constant across fitted values. It is detected visually by funnel/cone shapes in residual plots, or formally via statistical tests like the Breusch-Pagan test or White's test.

#### Q18: In simple linear regression, what is the exact algebraic relationship between $R^2$ and Pearson's $r$?
**Answer:** In simple univariate OLS with an intercept, $R^2 = r^2$, where $r$ is the sample Pearson correlation coefficient between $X$ and $Y$.

#### Q19: Why does the OLS regression line always pass through $(\bar{x}, \bar{y})$?
**Answer:** From the first-order condition $\frac{\partial S}{\partial b_0} = 0$, we have $\sum (y_i - b_0 - b_1 x_i) = 0 \implies \bar{y} = b_0 + b_1 \bar{x}$. Plugging in $x = \bar{x}$ yields $\hat{y} = \bar{y}$.

#### Q20: What is the sum of OLS residuals when an intercept is included?
**Answer:** Exactly zero: $\sum_{i=1}^n e_i = 0$.

---

### Advanced Tier

#### Q21: What is the Gauss-Markov Theorem?
**Answer:** The Gauss-Markov theorem states that under the assumptions of linearity, random sampling, strict exogeneity, homoscedasticity, and no autocorrelation, the OLS estimator is BLUE (Best Linear Unbiased Estimator)—it has the smallest variance among all unbiased linear estimators.

#### Q22: What happens to OLS estimates if the homoscedasticity assumption is violated?
**Answer:** The OLS parameter estimates $b_0$ and $b_1$ remain **unbiased and consistent**. However, they are no longer efficient (no longer BLUE), and the standard error formulas are biased, rendering standard hypothesis $t$-tests and confidence intervals invalid unless robust (Huber-White / sandwich) standard errors are used.

#### Q23: Why does regression not imply causation?
**Answer:** OLS minimizes geometric distances between data points; it possesses no mechanism to distinguish between:
1. $X$ causes $Y$
2. $Y$ causes $X$ (reverse causality)
3. An unmeasured confounder $Z$ causes both $X$ and $Y$
4. Spurious correlation from temporal trends or selection bias.

#### Q24: What is the impact of an extreme outlier on the regression slope?
**Answer:** If an outlier has high **leverage** (an extreme $x$-value) and is **discordant** (does not follow the linear trend), it exerts an outsized torque on the regression line, dramatically rotating the slope $b_1$ toward itself and deflating $R^2$.

#### Q25: What is the difference between interpolation and extrapolation?
**Answer:** Interpolation estimates $\hat{y}$ for input values within the range of observed training data ($\min(X) \le x_{\text{new}} \le \max(X)$). Extrapolation predicts for values outside this range ($x_{\text{new}} < \min(X)$ or $x_{\text{new}} > \max(X)$), where the assumed linear model may be completely invalid due to non-linear saturation or structural regime shifts.

#### Q26: What is the difference between a confidence interval and a prediction interval for $\hat{y}$?
**Answer:**
- A **confidence interval** bounds the uncertainty of the **mean response** $\mathbb{E}[Y | X = x^*]$: it reflects parameter estimation error and narrows as $n \to \infty$.
- A **prediction interval** bounds the uncertainty of an **individual new observation** $y^*$: it accounts for both parameter estimation error AND the inherent irreducible random error $\sigma^2$. A prediction interval is always significantly wider than a confidence interval.

#### Q27: How is slope $b_1$ related to standard deviations $s_X$ and $s_Y$?
**Answer:** $b_1 = r \cdot \frac{s_Y}{s_X}$. If both $X$ and $Y$ are standardized to $z$-scores ($s_X = s_Y = 1$), the slope equals Pearson's correlation coefficient ($b_1 = r$).

#### Q28: If you regress $X$ on $Y$ instead of $Y$ on $X$, is the new slope $b_1^* = \frac{1}{b_1}$?
**Answer:** No! Regressing $Y$ on $X$ yields $b_1 = r \frac{s_Y}{s_X}$, which minimizes vertical squared distances. Regressing $X$ on $Y$ yields $b_1^* = r \frac{s_X}{s_Y}$, which minimizes horizontal squared distances. Their product is $b_1 \cdot b_1^* = r^2$. They are reciprocals only if $|r| = 1$ (a perfect linear line).

#### Q29: What does a high training $R^2$ combined with a low test $R^2$ indicate?
**Answer:** It is the textbook signature of **overfitting**. The model has fit idiosyncrasies and noise in the training set that do not generalize to unseen test observations.

#### Q30: Why is RMSE always greater than or equal to MAE?
**Answer:** By Jensen's Inequality and the properties of quadratic means (Cauchy-Schwarz inequality), the square root of the mean of squares is mathematically bounded from below by the mean of absolute values ($\sqrt{\frac{1}{n}\sum e_i^2} \ge \frac{1}{n}\sum |e_i|$). Equality holds if and only if all absolute residuals are identical.

#### Q31: What is the bias-variance tradeoff in linear regression?
**Answer:** Prediction error decomposes into:
$$\text{Expected Test MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error } \sigma^2$$
Simple linear regression has high bias (rigid linear assumption) and low variance (stable estimates with low sensitivity to training perturbations).

#### Q32: What happens to $R^2$ if you add a non-zero constant $c$ to all $y$ values?
**Answer:** $R^2$ is completely unchanged. Adding a constant shifts both $y_i$ and $\bar{y}$ by $c$, so deviations $(y_i - \bar{y})$ and residuals $(y_i - \hat{y}_i)$ remain identical, preserving $SS_{\text{tot}}$ and $SS_{\text{res}}$.

#### Q33: How does Scikit-Learn's `LinearRegression().fit(X, y)` solve for parameters?
**Answer:** Scikit-learn solves the OLS problem using `scipy.linalg.lstsq`, which employs Singular Value Decomposition (SVD) of the centered feature matrix $X = U \Sigma V^T$. This approach is numerically stable even when $X$ is near-singular or contains multicollinear features.

#### Q34: What is the curse of extrapolation in business models?
**Answer:** Business systems operate under bounded economic regimes. If an advertising regression model trained on budgets up to ₹100,000 predicts sales for a ₹10,000,000 campaign, it ignores market saturation, channel exhaustion, and supply-chain constraints, producing catastrophically inflated revenue forecasts.

#### Q35: When should you fit a regression model without an intercept ($b_0 = 0$)?
**Answer:** Almost never, unless backed by rigorous physical first principles where $Y$ must strictly be zero when $X = 0$ (e.g., Ohm's Law $V = IR$). Forcing $b_0 = 0$ when the true intercept is non-zero biases $b_1$, causes $\sum e_i \neq 0$, and invalidates standard $R^2$ definitions.

---

## 💡 12. 10 Statistical Insights & Common Traps

1. **The Accuracy Fallacy:** Never report $R^2$ as an "accuracy percentage" to executives. State: *"Advertising expenditure accounts for 82% of the variance observed in sales revenue."*
2. **The "Zero Intercept" Blunder:** Do not suppress the intercept just because you believe zero advertising spend should yield zero sales. The baseline organic sales without advertising is a critical metric ($b_0$).
3. **The Extrapolation Trap:** Never feed feature values into `.predict()` that lie outside the minimum and maximum ranges of your training dataset without issuing an explicit extrapolation advisory.
4. **Outlier Magnetism:** A single discordant observation far out on the $X$-axis exerts immense leverage, dragging the regression slope toward itself like a gravitational magnet.
5. **Linearity Blindness:** If the true physical process is quadratic ($Y = X^2$), linear regression will fit a line through the parabola, reporting mediocre $R^2$ and masking a deterministic underlying pattern. Always plot residuals!
6. **The Unit-Scale Metric Trap:** MSE values in the millions (e.g., $150,000,000 \text{ Rs}^2$) look alarming to executives until converted to RMSE (₹12,247), which is readily understood relative to average sales of ₹200,000.
7. **The Data Leakage Risk:** Computing $\bar{x}$ or feature scalers on the entire dataset before `train_test_split` leaks future information into the training pipeline. Always split first, fit parameters on train only!
8. **Residual Autocorrelation in Time Series:** If your observations are sequential time periods (e.g., consecutive months), errors are often autocorrelated. Standard OLS underestimates standard errors, generating false confidence.
9. **Heteroscedastic Confidence Bands:** In the presence of heteroscedasticity, OLS predictions may be accurate on average, but prediction intervals will be too wide for small values and dangerously narrow for large values.
10. **The Causation Illusion:** A high $R^2$ (0.95) between marketing spend and sales does not prove marketing drove sales; seasonal holiday spikes could be driving both simultaneously.

---

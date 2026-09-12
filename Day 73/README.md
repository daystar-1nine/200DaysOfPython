# Day 73 — Sales Prediction Regression Engine

## 📊 Overview
A production-grade statistical and predictive machine learning system for single-variable linear modeling. The application explores the relationship between Advertising Spend and Sales Revenue using Ordinary Least Squares (OLS) regression, closed-form analytical derivations, gradient descent optimization, residual diagnostics (normality, zero mean, homoscedasticity), train/test split evaluation ($R^2$, MAE, MSE, RMSE), extrapolation boundary checking, business scenario projections, and 8 publication-grade visualizations.

---

## 🎯 Core Concepts Mastered
- **Linear Regression Foundations:** Supervised learning paradigm, feature ($X$) vs. target ($y$), relationship modeling vs. mere association.
- **Ordinary Least Squares (OLS):** Mathematical derivation minimizing the sum of squared errors ($SSE$), closed-form parameter solutions for slope ($\beta_1 = \frac{Cov(X, Y)}{Var(X)}$) and intercept ($\beta_0 = \bar{y} - \beta_1 \bar{x}$).
- **Gradient Descent Preview:** Iterative cost function optimization ($MSE$), partial derivatives $\frac{\partial J}{\partial \beta_0}$ and $\frac{\partial J}{\partial \beta_1}$, learning rate tuning, and parameter convergence.
- **Residual Diagnostics:** Verification of Gauss-Markov assumptions: zero mean ($E[e] = 0$), homoscedasticity (constant residual variance), normality of residuals, and absence of systematic patterns.
- **Model Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$).
- **Generalization & Validation:** 80/20 train/test splitting, out-of-sample performance verification, underfitting vs. overfitting detection.
- **Extrapolation Governance:** Identification and explicit flagging of out-of-domain budget scenarios to prevent false confidence beyond observed support $[X_{min}, X_{max}]$.

---

## 🏗️ Architecture & Modules

```text
Day 73/
├── Day73.md                              # Comprehensive Masterclass & 35 Technical Interview Q&As
├── README.md                             # Production Documentation & Run Guide
├── requirements.txt                      # Project Dependencies
├── pyproject.toml                        # Tool Configuration
├── pytest.ini                            # Test Runner Configuration
├── .gitignore                            # Cache and Artifact Exclusions
├── practice/
│   ├── task1_manual_regression_line.py   # Closed-form OLS parameter derivation from raw formulas
│   ├── task2_scikit_learn_regression.py  # Scikit-learn LinearRegression model fitting & verification
│   ├── task3_prediction_engine.py        # Scenario prediction function with extrapolation detection
│   ├── task4_residual_analysis.py        # Residual calculation, mean-zero proof, and variance test
│   ├── task5_r_squared_calculation.py    # Manual R-squared calculation vs sklearn.metrics.r2_score
│   ├── task6_regression_metrics.py       # Comprehensive MAE, MSE, RMSE, R-squared evaluation
│   └── task7_outlier_experiment.py       # High-leverage outlier perturbation impact experiment
├── coding_challenges/
│   ├── challenge1_manual_linear_regression.py # From-scratch OOP regressor matching scikit-learn
│   └── challenge2_gradient_descent_preview.py # Batch gradient descent engine with loss trajectory
├── data/
│   ├── raw/
│   │   └── advertising_sales.csv         # 600 advertising budget vs sales records
│   └── processed/
│       └── cleaned_sales.csv             # 598 sanitized, deduplicated, verified records
├── app/
│   ├── __init__.py                       # Package exports
│   ├── config.py                         # AppConfig dataclass & hyperparameter configurations
│   ├── loader.py                         # Robust CSV data ingestion with schema checks
│   ├── cleaner.py                        # Data validation, null handling, and domain filtering
│   ├── validator.py                      # Dimension, variance, and null checks
│   ├── eda.py                            # Summary statistics and Pearson correlation diagnostics
│   ├── regression.py                     # SimpleLinearRegressor wrapping scikit-learn & OLS math
│   ├── predictions.py                    # Scenario prediction generator with extrapolation flags
│   ├── metrics.py                        # MAE, MSE, RMSE, and R2 calculation engine
│   ├── residuals.py                      # Residual diagnostics (mean, variance ratio, skewness)
│   ├── visualizations.py                 # 8 publication-grade matplotlib charts (Agg backend)
│   ├── insights.py                       # Automated statistical insight generation
│   ├── report.py                         # ASCII summary report and CSV artifact exporters
│   └── main.py                           # CLI pipeline orchestrator
├── output/
│   ├── charts/                           # 8 publication-ready 300 DPI figures
│   │   ├── 01_raw_scatter.png
│   │   ├── 02_fitted_line.png
│   │   ├── 03_residuals_vs_fitted.png
│   │   ├── 04_residuals_histogram.png
│   │   ├── 05_residuals_qq.png
│   │   ├── 06_train_vs_test.png
│   │   ├── 07_actual_vs_predicted.png
│   │   └── 08_comprehensive_dashboard.png
│   ├── predictions.csv                   # Business scenario projections & extrapolation alerts
│   ├── regression_results.csv            # Train vs. test model performance benchmarks
│   └── regression_report.txt             # Executive summary ASCII report
└── tests/
    ├── conftest.py                       # Pytest fixtures and synthetic test arrays
    ├── test_loader.py                    # Loader schema, missing file, and edge-case tests
    ├── test_cleaner.py                   # Deduplication and positive value filtering tests
    ├── test_validator.py                 # Dataset validation and zero-variance guards
    ├── test_eda.py                       # Statistical summaries and correlation bounds
    ├── test_regression.py                # Regressor fitting, formulas, and no-intercept modes
    ├── test_predictions.py               # Monotonicity and extrapolation flag validation
    ├── test_metrics.py                   # MAE, MSE, RMSE, R2 mathematical precision
    ├── test_residuals.py                 # Mean zero, homoscedasticity, and skewness tests
    ├── test_insights.py                  # Insight synthesis and non-causal language rules
    └── test_integration.py              # End-to-end pipeline execution & artifact checks
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Complete Application Pipeline
Execute the main engine to ingest data, fit models, evaluate metrics, and export artifacts:
```bash
python app/main.py
```

### 3. Running Practice Tasks & Coding Challenges
```bash
python practice/task1_manual_regression_line.py
python practice/task2_scikit_learn_regression.py
python practice/task3_prediction_engine.py
python practice/task4_residual_analysis.py
python practice/task5_r_squared_calculation.py
python practice/task6_regression_metrics.py
python practice/task7_outlier_experiment.py

python coding_challenges/challenge1_manual_linear_regression.py
python coding_challenges/challenge2_gradient_descent_preview.py
```

### 4. Running Test Suite
Execute all 45 automated unit and integration tests:
```bash
pytest tests -v
```

---

## 📈 Performance Summary & Model Results

| Metric | Training Set (80%) | Testing Set (20%) | Overall Data |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | 0.8988 | 0.9022 | 0.8996 |
| **MAE** | 1,788.10 | 1,774.22 | 1,785.32 |
| **MSE** | 5,091,824.12 | 4,960,118.45 | 5,065,482.98 |
| **RMSE** | 2,256.51 | 2,227.13 | 2,250.66 |

### Fitted Regression Equation
$$	ext{Sales} = 0.4485 \times \text{Advertising\_Spend} + 12,410.82$$

- **Slope ($\beta_1$):** $0.4485$ (Every additional \$1,000 spent on advertising is associated with approximately \$448.50 in incremental sales revenue).
- **Intercept ($\beta_0$):** \$12,410.82 (Baseline sales revenue expected in the theoretical absence of advertising expenditure).
- **Residual Mean:** $-0.0000$ (Confirms unbiased estimator property).
- **Homoscedasticity Ratio:** $1.04$ (Homoscedastic, within $[0.5, 2.0]$ healthy threshold).

---

## 🛡️ Best Practices & Quality Standards
1. **Mathematical Equivalence:** Both closed-form OLS, Scikit-Learn `LinearRegression`, and batch gradient descent converge to identical optimal weights $(\beta_0, \beta_1)$.
2. **Deterministic Reproducibility:** Fixed random seeds across all data splits and gradient descent initializations.
3. **No Causal Fallacy:** Language strictly describes associations and predictive power without unjustified causal attribution.
4. **Safety Against Extrapolation:** Models explicitly detect when inputs lie outside observed sample bounds to warn decision makers against invalid linear extrapolation.

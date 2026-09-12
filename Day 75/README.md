# Day 75 — Advanced Regression Modeling Engine

![Progress](https://img.shields.io/badge/Day%2075%20%2F%20200-37.5%25%20Complete-blue)
![Days Remaining](https://img.shields.io/badge/125%20Days%20Remaining-orange)

## Overview
This project implements a robust Advanced Regression Modeling Engine to analyze non-linear relationships, address overfitting, and apply regularization techniques. We explore Polynomial Regression to capture complex patterns and Ridge/Lasso regularization to prevent overfitting by penalizing large coefficients.

## Concepts Covered
- Linear vs. Non-linear relationships
- Polynomial Regression
- Overfitting and Underfitting
- Bias-Variance Tradeoff
- Regularization ($L_1$ Lasso, $L_2$ Ridge, ElasticNet)
- Hyperparameter Tuning with GridSearchCV
- Data Leakage Prevention via Scikit-learn Pipelines

## Architecture Tree
```text
Day 75/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── cleaner.py
│   ├── validator.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── linear.py
│   │   ├── polynomial.py
│   │   ├── ridge.py
│   │   ├── lasso.py
│   │   └── elastic_net.py
│   ├── evaluation.py
│   ├── cross_validation.py
│   ├── tuning.py
│   ├── residuals.py
│   ├── coefficients.py
│   ├── model_comparison.py
│   ├── visualizations.py
│   ├── insights.py
│   ├── report.py
│   └── main.py
├── practice/
│   ├── task1_generate_nonlinear_data.py
│   ├── task2_linear_vs_polynomial.py
│   ├── task3_visualize_complexity.py
│   ├── task4_overfitting_experiment.py
│   ├── task5_ridge_vs_lasso.py
│   └── task6_cross_validation.py
├── coding_challenges/
│   ├── challenge1_model_selector.py
│   ├── challenge2_detect_overfitting.py
│   ├── challenge3_optimal_polynomial_degree.py
│   ├── challenge4_ridge_lasso_coefficients.py
│   └── challenge5_explain_model.py
├── data/
│   ├── raw/
│   │   └── advertising_sales.csv
│   └── processed/
│       └── cleaned_sales.csv
├── output/
│   ├── charts/
│   ├── coefficients.csv
│   ├── cv_results.csv
│   ├── hyperparam_results.csv
│   ├── model_results.csv
│   ├── predictions.csv
│   └── regression_report.txt
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_baseline.py
│   ├── test_cleaner.py
│   ├── test_cross_validation.py
│   ├── test_evaluation.py
│   ├── test_feature_engineering.py
│   ├── test_lasso.py
│   ├── test_linear.py
│   ├── test_loader.py
│   ├── test_model_comparison.py
│   ├── test_polynomial.py
│   ├── test_preprocessing.py
│   ├── test_residuals.py
│   ├── test_ridge.py
│   ├── test_tuning.py
│   └── test_validator.py
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── README.md
└── Day75.md
```

## Pipeline Diagram
```text
Raw Data (CSV)
  │
  ▼
[loader.py] ───► [cleaner.py] ───► [validator.py] ───► [feature_engineering.py]
                                                              │
                                                              ▼
                                                     [preprocessing.py] (ColumnTransformer)
                                                              │
        ┌───────────────────┬───────────────────┬─────────────┴─────┬───────────────────┐
        ▼                   ▼                   ▼                   ▼                   ▼
   [baseline.py]       [linear.py]      [polynomial.py]        [ridge.py]          [lasso.py]
        │                   │                   │                   │                   │
        └───────────────────┴───────────────────┼───────────────────┴───────────────────┘
                                                ▼
                                    [cross_validation.py]
                                                ▼
                                          [tuning.py] (GridSearchCV)
                                                ▼
                                     [model_comparison.py]
                                                │
                                                ├─► [evaluation.py] (MAE, MSE, RMSE, R²)
                                                ├─► [residuals.py] (Residual Diagnostics)
                                                ├─► [coefficients.py] (Sparsity & Shrinkage)
                                                ├─► [visualizations.py] (12 Diagnostic Charts)
                                                └─► [report.py & insights.py] (Executive Report)
```

## Tech Stack
- Python 3.10+
- Pandas, NumPy, SciPy
- Scikit-learn, Statsmodels
- Matplotlib, Seaborn
- Pytest

## Dataset Description
- **File:** `data/raw/advertising_sales.csv`
- **Rows:** 826 records
- **Features (16 columns):** `Record_ID`, `Date`, `TV_Spend`, `Digital_Spend`, `Radio_Spend`, `Discount`, `Quantity`, `Region`, `Category`, `Competitor_Price`, `Customer_Count`, `Advertising_Spend`, `Sales`, `Profit`, `Month`, `Year`

## Models Evaluated
1. **Baseline Model:** Constant empirical mean predictor.
2. **Multiple Linear Regression (OLS):** Closed-form linear plane estimator.
3. **Polynomial Regression (Degree 2):** Quadratic basis expansion with interaction terms.
4. **Polynomial Regression (Degree 3):** Cubic basis expansion modeling non-linear inflection points.
5. **Ridge Regression ($L_2$):** Penalized shrinkage addressing multicollinear channels.
6. **Lasso Regression ($L_1$):** Sparse parameter estimation with automated feature selection.
7. **Elastic Net ($L_1 + L_2$):** Balanced sparsity and grouped correlation handling.

## Visualizations
Saved in `output/charts/`:
1. `1_actual_vs_pred.png` — Actual vs. Predicted values along the $45^\circ$ diagonal.
2. `2_residual_plot.png` — Fitted values vs. residuals assessing homoscedasticity.
3. `3_residual_dist.png` — Histogram with KDE verifying residual normality.
4. `4_deg_train_err.png` — Polynomial degree vs. training RMSE curve.
5. `5_deg_test_err.png` — Polynomial degree vs. test RMSE curve (overfitting boundary).
6. `6_alpha_cv.png` — Regularization strength ($\alpha$) vs. Cross-Validated RMSE.
7. `7_model_comp.png` — Model comparison bar chart across RMSE.
8. `8_corr.png` — Pairwise correlation heatmap across numeric features.
9. `9_coef_comp.png` — Direct coefficient comparison between Linear, Ridge, and Lasso.
10. `10_adv_sales.png` — Total advertising spend vs. Sales with non-linear trendline.
11. `11_month_sales.png` — Monthly seasonal sales trajectory.
12. `12_complexity_err.png` — Dual-curve model complexity vs. training/test error.

## Testing
Comprehensive test suite containing **65 automated unit and integration tests** passing with 100% success:
- Data validation, loader error handling, cleaning, and feature engineering
- Model fitting, prediction shape, and property contracts across all 6 estimators
- Evaluation metrics ($\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $R^2$, $\text{Adjusted } R^2$)
- $K$-Fold cross-validation and hyperparameter grid tuning (`GridSearchCV`)
- Residual statistics, coefficient shrinkage, and overfitting detection

## Running the Project
```bash
# Navigate to Day 75
cd "Day 75"

# Run the complete end-to-end ML pipeline
python app/main.py

# Run all 65 automated unit tests
pytest tests/ -v
```

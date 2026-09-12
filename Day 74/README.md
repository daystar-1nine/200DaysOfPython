# Day 74 — Multiple Linear Regression: Advanced Sales Prediction Engine

![Progress](https://img.shields.io/badge/Day%2074%20%2F%20200-37%25%20Complete-blue)

## Overview
This project builds an advanced sales prediction engine using Multiple Linear Regression. It demonstrates how to handle multiple independent variables to predict a continuous target variable, featuring data preprocessing, feature engineering, multicollinearity checks, and performance evaluation.

## Concepts Covered
- Multiple Linear Regression Equations
- Data Preprocessing and One-Hot Encoding
- Handling Multicollinearity (VIF)
- Feature Scaling
- Evaluating Model Metrics (R², Adjusted R², MAE, RMSE)
- Residual Analysis

## Architecture
```text
Day 74/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── cleaner.py
│   ├── validator.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── regression.py
│   ├── predictions.py
│   ├── metrics.py
│   ├── multicollinearity.py
│   ├── residuals.py
│   ├── feature_analysis.py
│   ├── visualizations.py
│   ├── insights.py
│   ├── report.py
│   └── main.py
├── coding_challenges/
│   ├── challenge1_build_without_tutorial.py
│   ├── challenge2_feature_removal.py
│   ├── challenge3_multicollinearity_experiment.py
│   └── challenge4_budget_allocation.py
├── practice/
│   ├── task1_manual_prediction.py
│   ├── task2_multiple_regression_demo.py
│   ├── task3_categorical_encoding.py
│   ├── task4_model_comparison.py
│   ├── task5_multicollinearity_vif.py
│   └── task6_residual_diagnostics.py
├── data/
│   ├── raw/
│   │   └── advertising_sales.csv
│   └── processed/
│       └── cleaned_sales.csv
├── output/
│   ├── charts/
│   ├── coefficients.csv
│   ├── model_metrics.csv
│   ├── predictions.csv
│   ├── regression_report.txt
│   └── vif_report.csv
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cleaner.py
│   ├── test_feature_engineering.py
│   ├── test_insights.py
│   ├── test_loader.py
│   ├── test_metrics.py
│   ├── test_multicollinearity.py
│   ├── test_predictions.py
│   ├── test_preprocessing.py
│   ├── test_regression.py
│   ├── test_residuals.py
│   └── test_validator.py
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── README.md
└── Day74.md
```

## Pipeline Diagram
```text
Raw Data (CSV)
  │
  ▼
[loader.py] ───► [cleaner.py] ───► [validator.py] ───► [feature_engineering.py]
                                                              │
                                                              ▼
[predictions.py] ◄─── [regression.py] ◄─── [preprocessing.py] (ColumnTransformer)
       │
       ├─► [metrics.py] (R², Adj R², RMSE, MAE)
       ├─► [residuals.py] (Diagnostics, Homoscedasticity)
       ├─► [multicollinearity.py] (VIF Calculation)
       ├─► [visualizations.py] (10 Diagnostic Charts)
       └─► [report.py & insights.py] (Executive Business Report)
```

## Tech Stack
- Python 3.10+
- Pandas & NumPy
- Scikit-Learn
- Statsmodels
- Matplotlib & Seaborn
- Pytest

## Dataset
Located at `data/raw/advertising_sales.csv`. 
- **Rows:** 826
- **Columns:** 16 (`Record_ID`, `Date`, `TV_Spend`, `Digital_Spend`, `Radio_Spend`, `Discount`, `Quantity`, `Region`, `Category`, `Competitor_Price`, `Customer_Count`, `Advertising_Spend`, `Sales`, `Profit`, `Month`, `Year`)

## Model Performance
| Metric | Value |
|:---|:---|
| **$R^2$** | **0.9144** |
| **Adjusted $R^2$** | **0.9076** |
| **MAE** | **₹1,179.80** |
| **RMSE** | **₹1,459.70** |
| **MSE** | **2,130,716.23** |
| **Samples ($n$)** | **163** (Test Set) |
| **Features ($p$)** | **12** (Encoded Matrix) |

## Visualizations
Saved in `output/charts/`:
1. `actual_vs_predicted.png` — Scatter alignment against the $45^\circ$ reference line.
2. `residuals_vs_fitted.png` — Homoscedasticity and zero-mean check.
3. `residual_distribution.png` — Histogram with Kernel Density Estimate (KDE).
4. `correlation_heatmap.png` — Feature correlation matrix.
5. `tv_spend_vs_sales.png` — Univariate relationship and regression fit.
6. `coefficients.png` — Standardized parameter weights.
7. `sales_by_region.png` — Regional sales boxplot distribution.
8. `monthly_sales.png` — Time-series seasonality tracking.
9. `model_comparison.png` — Progressive feature addition comparison.
10. `vif_chart.png` — Variance Inflation Factor diagnostics with threshold line.

## Testing
Contains **53 automated unit and integration tests** passing with 100% success:
- Data validation, loader error boundaries, and cleaning
- Feature engineering derived calculations
- Model fitting, coefficient extraction, and pipeline transformations
- Metric calculations (MAE, MSE, RMSE, $R^2$, Adjusted $R^2$)
- Multicollinearity detection and VIF scoring
- Residual diagnostics and non-causal insight generation

## Running the Project
```bash
# Navigate to Day 74
cd "Day 74"

# Run the complete end-to-end ML pipeline
python app/main.py

# Run all 53 automated unit tests
pytest tests/ -v
```

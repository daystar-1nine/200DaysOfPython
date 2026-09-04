# 🐍 Day 59 — Advanced Pandas: Transform, Rank, Rolling & EDA Foundations

## 🏆 Overview
Welcome to **Day 59** of the 200 Days of Python Challenge. Today moves from data cleaning and manipulation into **Exploratory Data Analysis (EDA)**. You will master row-aligned group metrics (`transform()`), intra-group rankings (`rank()`), time-series operations (`shift()`, `diff()`, `pct_change()`), moving averages (`rolling()`), quantiles, IQR outlier detection, correlation/covariance, and building a full **E-Commerce Exploratory Data Analysis Engine**.

---

## 🎯 Topics Learned
- `groupby().transform()` for broadcast row alignment without reducing rows.
- `groupby().rank()` for rank ordering within category/group partitions.
- Time-series lag analysis with `shift()`, `diff()`, and percentage growth `pct_change()`.
- Moving window aggregations with `rolling()` (rolling mean, rolling sum).
- Cumulative sequence tracking using `cumsum()` and `cumcount()`.
- Distribution analysis via percentiles, quantiles, and IQR (Interquartile Range) outlier bounds.
- Bivariate correlation matrices (`corr()`) and covariance (`cov()`).
- 12-Phase Exploratory Data Analysis (EDA) framework and business insight extraction.

---

## 📂 Project Structure
```text
Day 59/
├── Day59.md                   # Masterclass notes & 30 Technical Interview Q&As
├── exercises/                 # 6 Standalone Practical Tasks
│   ├── task1_transform.py
│   ├── task2_ranking.py
│   ├── task3_shift.py
│   ├── task4_rolling.py
│   ├── task5_outliers.py
│   └── task6_correlation.py
├── coding_challenges/         # 5 Standalone Coding Challenges
│   ├── challenge1_category_transform.py
│   ├── challenge2_department_rankings.py
│   ├── challenge3_time_series_growth.py
│   ├── challenge4_iqr_outliers.py
│   └── challenge5_eda_summary_function.py
├── data/
│   ├── raw/
│   │   └── ecommerce_sales.csv # 350+ record synthetic transactional sales dataset
│   └── processed/
│       └── cleaned_sales.csv   # Enriched & cleaned dataset with derived metrics
├── app/                       # E-Commerce EDA Engine Package
│   ├── __init__.py
│   ├── main.py                # 12-Phase EDA Pipeline Entry Point
│   ├── loader.py              # Data ingestion & schema validation
│   ├── cleaner.py             # Data clean-up & datetime parsing
│   ├── transformer.py         # Financial metric calculation (Revenue, Cost, Profit) & date features
│   ├── descriptive.py         # Descriptive statistics, quantiles & IQR metrics
│   ├── group_analysis.py      # Regional, Category, Product, Customer aggregations & rankings
│   ├── time_analysis.py       # Time-series analytics, MoM growth & rolling averages
│   ├── correlation.py         # Numerical correlation & covariance matrices
│   ├── outliers.py            # IQR outlier bounds detection & filtering
│   └── report.py              # ASCII EDA Report generator (10+ Insights) & CSV Exporter
├── output/                    # Generated EDA artifacts & CSV summaries
│   ├── eda_report.txt
│   ├── regional_analysis.csv
│   ├── category_analysis.csv
│   ├── customer_analysis.csv
│   ├── product_analysis.csv
│   └── monthly_analysis.csv
├── tests/                     # 25+ Pytest Automation Test Cases
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_transformer.py
│   ├── test_group_analysis.py
│   ├── test_time_analysis.py
│   └── test_outliers.py
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Running the Project

### 1. Execute Main EDA Pipeline
```bash
python Day 59/app/main.py
```

### 2. Run Automated Pytest Suite
```bash
pytest "Day 59/tests/"
```

# Day 60 — Business Intelligence Analytics Engine

## 200 Days of Python

Day 60 milestone project combining:

- Python (Type Hints, Dataclasses, Enums, Pathlib, Collections, JSON/CSV)
- NumPy (Vectorization, Broadcasting, NaN Handling, Matrix Algebra)
- Pandas (Series, DataFrames, loc/iloc, Groupby, Aggregations, Merges)
- Data Cleaning (String Normalization, Numeric Coercion, Imputation, Deduplication)
- Data Processing (Safe Type Casting, Validation Rules, Model Pipelines)
- Exploratory Data Analysis (Quantiles, IQR Outliers, Correlation, Transform, Rank, Rolling)

## Project Overview

A reusable, production-grade business analytics engine that processes raw e-commerce transaction data, audits data quality, performs multidimensional analytics, and generates dynamic business insights and analytical reports.

## Pipeline Architecture

```text
Raw Data (CSV)
      ↓
Data Ingestion (loader.py)
      ↓
Data Cleaning & Normalization (cleaner.py)
      ↓
Business Rule Validation (validator.py)
      ↓
Financial Transformation & Temporal Engineering (transformer.py)
      ↓
Multidimensional Analysis Engine (analysis/)
  ├── Macro Overview
  ├── Customer Performance & Rankings
  ├── Product & Category Metrics
  ├── Regional Analysis
  ├── Time-Series Trends & Rolling Windows
  ├── Descriptive Statistics & Correlations
  └── IQR Outlier Auditing
      ↓
Automated Dynamic Business Insights (insights.py)
      ↓
Executive Reports & CSV Exports (reports.py)
```

## Features

- Robust CSV ingestion with schema validation
- Automated data cleaning, date coercion, and deduplication
- Business rule domain boundary validation
- Revenue, Cost, Profit, and Profit Margin derivations
- Customer spend rankings, order counts, and Average Order Value (AOV)
- Product performance and intra-category rankings
- Regional sales, profit, and order volume breakdowns
- Category revenue, profit, quantity, and discount comparisons
- Monthly trend tracking, Month-over-Month (MoM) growth rates, and 3-month rolling averages
- Continuous variable Pearson correlation and covariance matrices
- Interquartile Range (IQR) statistical outlier auditing
- Automated, data-driven business insight generation (10+ insights)
- Multi-format artifact generation (Executive ASCII report, Data Quality Audit, 5 CSV summaries)

## Testing Target

30+ automated unit tests implemented with Pytest across all pipeline components.

## Technologies

- Python 3.10+
- NumPy
- Pandas
- Pytest

## Progress

60 / 200 Days Completed (30.0%) — **MILESTONE 2 ACHIEVED!** 🏆

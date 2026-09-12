# Day 69 — Confidence Intervals & Statistical Estimation Analyzer

## Overview
A production-grade statistical inference and estimation application for computing point and interval estimates:
- Point Estimation ($\bar{x}, \hat{p}$)
- Margin of Error ($ME = z_{\alpha/2} \cdot SE$ or $t_{\alpha/2, df} \cdot SE$)
- Z-Distribution (known $\sigma$) vs T-Distribution (unknown $\sigma$, heavy tails)
- Single-Sample Confidence Intervals for Population Mean
- Confidence Intervals for Population Proportion (Wald & Wilson Score)
- Sample Size Determination for targeted precision
- Non-Parametric Bootstrap Confidence Intervals (percentile method)
- 7 publication-grade diagnostic charts
- 35+ automated Pytest unit and integration tests

## Project Structure
```text
Day 69/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── validator.py
│   ├── loader.py
│   ├── stats_calc.py
│   ├── estimators.py
│   ├── confidence_intervals.py
│   ├── sample_size.py
│   ├── bootstrap.py
│   ├── insights.py
│   ├── visualizations.py
│   ├── report.py
│   └── main.py
├── practice/
├── coding_challenges/
├── data/
│   └── customer_orders.csv
├── output/
│   ├── charts/
│   ├── confidence_intervals.csv
│   ├── bootstrap_results.csv
│   └── estimation_report.txt
└── tests/
```

## Running the Application
```bash
python -m app.main
```

## Running Tests
```bash
pytest tests -v
```

## Progress
- **Day 69 / 200 Days Completed (34.5%)**

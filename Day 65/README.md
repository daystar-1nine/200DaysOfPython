# Day 65 — Statistics for Data Science: Descriptive Statistics

## 🎯 Overview
Day 65 focuses on the mathematical and statistical machinery powering data science, visual analytics, and exploratory data analysis. We explore central tendency, dispersion, percentiles, Tukey's IQR outlier fences, Z-score standardization, sample vs population variance (`ddof`), skewness, kurtosis, and build a reusable **Statistical Analysis Engine**.

## 📊 Core Topics Covered
- **Central Tendency:** Arithmetic Mean, Median (even/odd), Mode (unimodal, multimodal, uniform).
- **Dispersion:** Range, Variance ($\sigma^2$ vs $s^2$, Bessel's correction `ddof=1`), Standard Deviation.
- **Percentiles & Quartiles:** P10 through P99, Five-number summary ($Q_1, Q_2, Q_3$).
- **Outlier Detection:** IQR boundaries ($Q_1 - 1.5 \times IQR$, $Q_3 + 1.5 \times IQR$), Z-score thresholding ($|z| > 3$).
- **Distribution Diagnostics:** Pearson's skewness, Fisher's kurtosis (excess kurtosis, heavy tails vs light tails).
- **Edge Cases:** Empty series, constant data, NaN handling, infinite values, single-element arrays.

## 📁 Directory Structure
```text
Day 65/
├── Day65.md                   # Masterclass Notes, 25 Interview Q&As & Assessment Solutions
├── app/                       # Main Capstone: Statistical Analysis Engine
│   ├── config.py              # Central Paths & Statistical Thresholds
│   ├── loader.py              # Resilient Dataset Loader
│   ├── validator.py           # Domain & Edge Case Validation
│   ├── statistics/            # Decoupled Statistical Computation Engines
│   │   ├── central_tendency.py
│   │   ├── dispersion.py
│   │   ├── percentiles.py
│   │   ├── outliers.py
│   │   ├── zscore.py
│   │   └── distribution.py
│   ├── visualizations.py      # Statistical Plot Generators (Agg Headless)
│   ├── insights.py            # Automated Business Intelligence Insights
│   ├── report.py              # Multi-Report ASCII & CSV Exporter
│   └── main.py                # Production Pipeline CLI Entry Point
├── exercises/                 # 10 Practical Hands-on Tasks
├── coding_challenges/         # 5 Advanced Statistical Challenges
├── data/                      # E-Commerce Sales Dataset (750 rows)
├── output/                    # Generated Reports & Visualizations
│   ├── charts/                # Publication-Grade Diagnostic Figures
│   ├── statistical_summary.csv
│   ├── percentile_report.csv
│   ├── outlier_report.csv
│   ├── zscore_report.csv
│   └── statistical_report.txt
├── tests/                     # 30+ Pytest Automation Test Cases
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🚀 How to Run
```bash
# Run tests
pytest

# Run the Statistical Analysis Engine
python -m app.main
```

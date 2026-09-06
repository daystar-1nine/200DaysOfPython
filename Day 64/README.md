# Day 64 — Advanced Seaborn & Statistical EDA

## 📊 Overview
Day 64 marks the transition from individual visualizations to **multidimensional statistical exploratory data analysis (EDA)**. By leveraging Seaborn figure-level plotting functions (`catplot()`, `relplot()`), small multiples faceting (`col`, `row`), multidimensional encoding (`hue`, `size`, `style`), robust central tendency estimators (`mean` vs `median`), and uncertainty modeling (`errorbar`, bootstrapped confidence intervals), we investigate complex business relationships across multiple variables simultaneously.

We apply these principles to build an enterprise-grade **Advanced Customer Analytics Report Engine** that parses, profiles, and visualizes 750+ multi-channel transactions while strictly separating business calculations from chart rendering routines.

---

## 🎯 Topics Covered
- **Figure-Level vs Axes-Level Architectures**:
  - `catplot()` for categorical distribution and summary matrices.
  - `relplot()` for continuous bivariate and multivariate relationships.
  - Figure-level FacetGrid management, aspect ratios, and dimension scaling.
- **Multidimensional Faceting & Small Multiples**:
  - Partitioning complex relationships across grid columns (`col`) and rows (`row`).
  - Preventing cognitive overload in facet layouts.
  - Uncovering subgroup variations obscured by macro aggregations (Simpson's Paradox).
- **Statistical Aggregations & Estimators**:
  - Parametric `estimator="mean"` vs non-parametric `estimator="median"`.
  - The impact of skewness and heavy tails on group comparisons.
  - Bootstrapped confidence intervals (`errorbar=("ci", 95)`), standard error (`"se"`), and variance.
- **Categorical & Distributional Comparison**:
  - When to choose Bar vs Box vs Violin vs Strip vs Swarm plots.
  - Outlier detection via the Tukey 1.5 * IQR rule.
  - Multimodality and density shape inspection with mirrored KDE violins.
- **Time & Relational Dynamics**:
  - Longitudinal multi-series trendlines with confidence intervals.
  - Ordinary Least Squares (OLS) linear regressions and residual analysis.
  - Lower-triangle masked Pearson correlation matrices.
- **Clean Architecture & Separation of Concerns**:
  - Strict architectural decoupling: Business analytics functions compute data; visualization functions strictly receive processed DataFrames and render plots.

---

## 📂 Repository Structure
```text
Day 64/
├── Day64.md                        # Masterclass Documentation, 20 Technical Q&As & Assessment Solutions
├── pyproject.toml                  # Pytest Configuration
├── requirements.txt                # Dependencies (pandas, numpy, matplotlib, seaborn, pytest)
├── README.md                       # Comprehensive Guide & Architecture
├── .gitignore                      # Git Ignore Rules
├── data/
│   └── ecommerce_sales.csv         # Enterprise dataset (750 transactions, 19 columns)
├── exercises/
│   ├── task1_grouped_barplot.py    # Grouped barplot: Regional revenue with Category hue
│   ├── task2_boxplot.py            # Regional profit distributions & IQR outliers
│   ├── task3_violinplot.py         # Category revenue density profiles & multimodality
│   ├── task4_stripplot.py          # Observation-level strip plot vs boxplot comparison
│   ├── task5_faceting.py           # Faceted catplot of revenue across product categories
│   ├── task6_relplot_multivariate.py # 4D relplot: Revenue, Profit, Region (hue), Quantity (size)
│   ├── task7_faceted_relationship.py# Category-faceted relplot testing slope consistency
│   ├── task8_mean_vs_median.py     # Side-by-side comparative barplots of Mean vs Median
│   └── task9_correlation_heatmap.py# Masked lower-triangle Pearson correlation matrix
├── coding_challenges/
│   ├── challenge1_faceted_analysis.py   # Revenue vs Profit faceted by Category & colored by Region
│   ├── challenge2_mean_vs_median.py     # Quantifying divergence between Mean and Median
│   ├── challenge3_outlier_investigation.py # Root-cause analysis of top 1% highest revenue transactions
│   ├── challenge4_simpsons_paradox.py   # Empirical subgroup vs aggregate trend divergence test
│   └── challenge5_auto_insights_generator.py # Automated dynamic natural-language business insight engine
├── app/
│   ├── __init__.py
│   ├── config.py                   # Central paths, dimensions, palettes & themes
│   ├── loader.py                   # Safe dataset loader and schema verification
│   ├── cleaner.py                  # Preprocessing, type coercion & segment mapping
│   ├── analyzer.py                 # Pure business analytics (0 visual dependencies)
│   ├── distributions.py            # Distribution plots (Revenue, Profit, Quantity)
│   ├── categorical.py              # Categorical plots (Boxplots, Violins, Barplots, Countplots)
│   ├── relationships.py            # Relational plots (catplot hue/faceted, relplot scatter/faceted)
│   ├── time_analysis.py            # Multi-series longitudinal lineplots
│   ├── correlation.py              # Masked correlation heatmap generator
│   ├── customer_analysis.py        # Top customer horizontal bar charts
│   ├── visualizations.py           # Master coordinator generating all 17 charts
│   ├── report.py                   # Author of 9-section report with 15+ business insights
│   └── main.py                     # CLI pipeline orchestrator
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test fixtures and synthetic data
│   ├── test_loader.py              # Dataset ingestion tests
│   ├── test_cleaner.py             # Preprocessing & cleaning tests
│   ├── test_analyzer.py            # Analytics & metric calculation tests
│   ├── test_distributions.py       # Distribution chart tests
│   ├── test_categorical.py         # Categorical chart tests
│   ├── test_relationships.py       # Relational & faceted chart tests
│   ├── test_time_analysis.py       # Time-series trend chart tests
│   ├── test_correlation.py         # Correlation matrix & heatmap tests
│   └── test_customer_analysis.py   # Top customer ranking chart tests
└── output/
    ├── charts/                     # 17 Publication-grade statistical figures (300 DPI)
    └── advanced_eda_report.txt     # 9-Section comprehensive executive report with 15+ insights
```

---

## 🚀 Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Automated Tests
```bash
pytest tests/ -v
```

### 3. Run the Main Analytics Pipeline
```bash
python -m app.main
```
Generates:
- All 17 statistical charts in `output/charts/`
- The full 9-section executive report in `output/advanced_eda_report.txt`

### 4. Run Exercises & Coding Challenges
```bash
python exercises/task1_grouped_barplot.py
python coding_challenges/challenge5_auto_insights_generator.py
```

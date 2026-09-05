# Day 63 — Seaborn: Statistical Data Visualization

## 📊 Overview
Day 63 marks the transition into **Statistical Data Visualization**. While Matplotlib provides low-level, pixel-precise graphic primitives, **Seaborn** operates at a higher statistical abstraction. Built on top of Matplotlib and integrated with Pandas DataFrames, Seaborn automates complex statistical aggregations, confidence interval estimation, distribution modeling (KDE, ECDF), multivariate color/size/style encodings, and relational grids.

Through Day 63, we build an enterprise-grade **Customer & Sales EDA Visualization Engine** that extracts deep empirical insights from over 750 multi-channel e-commerce transactions.

---

## 🎯 Topics Covered
- **Seaborn Architecture & Core Philosophy**:
  - Matplotlib foundation vs Seaborn high-level statistical interface.
  - Tidy (long-form) data structures and automated DataFrame column mapping.
  - Axes-level (`histplot`, `boxplot`, `scatterplot`) vs Figure-level (`displot`, `catplot`, `relplot`) functions.
- **Univariate Distribution Modeling**:
  - Histogram density estimation with `sns.histplot(kde=True)`.
  - Kernel Density Estimation (`sns.kdeplot(fill=True)`).
  - Empirical Cumulative Distribution Functions (`sns.ecdfplot()`).
- **Categorical & Group Comparison**:
  - Outlier detection and quartile anatomy via `sns.boxplot()`.
  - Density shape and multimodality comparison via `sns.violinplot()`.
  - Summary aggregation with bootstrapped confidence intervals via `sns.barplot()`.
  - Frequency distribution across subsets with `sns.countplot()`.
  - Observation-level inspection with `sns.stripplot()` and `sns.swarmplot()`.
- **Multivariate Relational Analysis**:
  - Multi-dimensional scatter encodings: `hue` (category), `size` (magnitude), `style` (marker).
  - Time-series aggregation with variance bands via `sns.lineplot()`.
  - Linear regression trends and confidence bands via `sns.regplot()` and `sns.lmplot()`.
- **Correlation & Matrix Visualization**:
  - Pearson correlation matrices visualized through `sns.heatmap()`.
  - Upper-triangle masking (`np.triu()`) to eliminate visual redundancy.
  - Diverging colormaps (`coolwarm`, `vlag`) centered at zero.
- **Multi-Plot Grids & Small Multiples**:
  - All-vs-all metric exploration with `sns.pairplot(corner=True, hue=...)`.
  - Bivariate + Univariate marginal diagnostics via `sns.jointplot()`.
  - Conditional facet subplots via `sns.FacetGrid`.
- **Aesthetic Customization & Production Standards**:
  - Theme control: `sns.set_theme()`, `sns.set_style("whitegrid")`, `sns.set_context("notebook")`.
  - Spines reduction using `sns.despine()`.
  - Memory hygiene: strictly enforcing `plt.close(fig)` across all rendering pipelines.

---

## 📂 Repository Structure
```text
Day 63/
├── Day63.md                        # Masterclass Documentation, 20 Technical Q&As & Mini Assessment
├── pyproject.toml                  # Pytest Configuration (strict markers, auto-discovery)
├── requirements.txt                # Dependencies (numpy, pandas, matplotlib, seaborn, pytest)
├── README.md                       # Day 63 Comprehensive Guide
├── .gitignore                      # Git ignore rules
├── data/
│   └── ecommerce_sales.csv         # Enterprise dataset (750 transactions, 19 columns)
├── exercises/
│   ├── task1_histplot_kde.py       # Revenue distribution with KDE & mean/median markers
│   ├── task2_boxplot.py            # Category order values & IQR outlier visualization
│   ├── task3_violinplot.py         # Regional revenue density & inner quartile distributions
│   ├── task4_countplot.py          # Category order counts partitioned by customer segment
│   ├── task5_scatterplot_hue.py     # Revenue vs Profit multivariate encoding with regression
│   ├── task6_correlation_heatmap.py# Masked lower-triangle Pearson correlation matrix
│   ├── task7_pairplot.py           # Pairwise matrix across financial metrics by segment
│   └── task8_sales_quick_eda.py    # 4-panel quick EDA multi-chart analytical figure
├── coding_challenges/
│   ├── challenge1_marks_hist_kde.py    # Academic examination marks normality & KDE analysis
│   ├── challenge2_dept_box_violin.py   # Departmental salary distributions (Box + Violin)
│   ├── challenge3_revenue_profit_hue.py# Revenue vs Profit by Segment & Region
│   ├── challenge4_correlation_extremes.py# Dynamic identification of extreme correlation pairs
│   └── challenge5_combined_eda_figure.py # 6-panel comprehensive publication-grade EDA figure
├── app/
│   ├── __init__.py
│   ├── config.py                   # Central themes, color palettes, DPI, output directories
│   ├── loader.py                   # Safe dataset ingestion and schema validation
│   ├── cleaner.py                  # Type conversion, date normalization, segment derivation
│   ├── analyzer.py                 # Pure statistical computations (skewness, IQR, correlation)
│   ├── distributions.py            # Univariate distribution charts (hist, KDE, box, violin)
│   ├── categorical.py              # Categorical comparison charts (bar, count, strip, swarm)
│   ├── relationships.py            # Bivariate and multivariate plots (scatter, line, regplot)
│   ├── correlation.py              # Masked heatmap matrix and top correlation extractors
│   ├── charts.py                   # High-level pipeline orchestrating all 12 publication charts
│   ├── report.py                   # 10-Question Statistical Executive Report generator
│   └── main.py                     # CLI orchestrator generating charts and executive report
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Synthetic DataFrame fixtures and test environment
│   ├── test_loader.py              # Tests for dataset loader and schema constraints
│   ├── test_cleaner.py             # Tests for cleaner and deterministic transformations
│   ├── test_analyzer.py            # Tests for pure statistical metrics and correlation math
│   ├── test_distributions.py       # Tests for distribution chart renderers
│   ├── test_categorical.py         # Tests for categorical chart renderers
│   ├── test_relationships.py       # Tests for relational and regression chart renderers
│   └── test_correlation.py         # Tests for heatmap generation and matrix masking
└── output/
    ├── charts/                     # 12 Publication-grade statistical charts (300 DPI)
    └── eda_visualization_report.txt# Comprehensive 10-Question Statistical Executive Report
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Automated Tests
```bash
pytest tests/ -v
```

### 3. Run the Capstone EDA Visualization Engine
```bash
python -m app.main
```
This generates:
- 12 High-resolution (300 DPI) statistical visualizations in `output/charts/`.
- The comprehensive executive text report in `output/eda_visualization_report.txt`.

### 4. Run Exercises & Challenges
```bash
python exercises/task1_histplot_kde.py
python coding_challenges/challenge5_combined_eda_figure.py
```

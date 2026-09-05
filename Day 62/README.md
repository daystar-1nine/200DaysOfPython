# Day 62 — Professional Matplotlib & Dashboard-Style Visualization

## 📊 Overview
Day 62 elevates data visualization from standalone charts to **professional multi-panel analytical dashboards**. By integrating Figure & Axes management, custom layout specifications with `GridSpec`, precision formatting, automated KPI card generation, and domain-informed annotations, we synthesize complex e-commerce metrics into a cohesive, publication-quality executive report.

## 🎯 Topics Covered
- **Figure & Axes Architecture**: Deep dive into multi-panel layouts, coordinate domains, and container relationships.
- **Multiple Subplots**: Grid coordination with `plt.subplots(nrows, ncols)` vs procedural `plt.subplot()`.
- **Advanced Grid Layouts with `GridSpec`**: Complex asymmetric layouts spanning multiple rows and columns.
- **Precision Axis Customization**:
  - Setting deliberate axis limits (`set_xlim`, `set_ylim`) without distorting proportions.
  - Tick positioning (`set_xticks`) and rotation (`tick_params(axis="x", rotation=...)`).
  - Currency and number formatting via `matplotlib.ticker.FuncFormatter`.
  - Date formatting and locator rules with `matplotlib.dates` (`DateFormatter`, `MonthLocator`).
- **Contextual Annotations & Reference Lines**:
  - Horizontal reference lines (`axhline`) for corporate targets and metric averages.
  - Vertical reference lines (`axvline`) for deadlines, cutoffs, and thresholds.
  - Automated peak annotations (`ax.annotate`) identifying data maxima dynamically.
- **Dual-Axis Charts (`twinx()`)**: Synchronizing related metrics with distinct unit scales (e.g., Revenue vs Profit Margin).
- **Static Executive KPI Cards**: Creating clean, text-driven metric tiles.
- **Visual Hierarchy & Dashboard Storytelling**: Guiding executive attention from high-level KPIs down to category and SKU-level diagnostics.

## 📂 Repository Structure
```text
Day 62/
├── Day62.md                        # Masterclass Notes, 20 Interview Q&As & Revision Test Solutions
├── pyproject.toml                  # Pytest Configuration
├── requirements.txt                # Dependencies (pandas, numpy, matplotlib, pytest)
├── README.md                       # Comprehensive Documentation
├── .gitignore                      # Ignore Rules
├── exercises/
│   ├── task1_multiple_subplots.py  # 2x2 multi-panel layout (Line, Bar, Scatter, Hist)
│   ├── task2_reference_line.py     # Average sales horizontal reference line
│   ├── task3_auto_annotation.py    # Automated dynamic peak annotation
│   ├── task4_currency_format.py    # Indian Rupee (₹) axis formatting with FuncFormatter
│   ├── task5_date_formatting.py    # 12-month date formatting with mdates
│   └── task6_gridspec_layout.py    # Asymmetric 1-large + 3-small GridSpec layout
├── coding_challenges/
│   ├── challenge1_target_comparison.py # Actual vs Target revenue with reference line
│   ├── challenge2_dynamic_peak_annot.py# Automated peak month detection & callout
│   ├── challenge3_kpi_dashboard.py     # 4-card KPI metric header dashboard
│   ├── challenge4_dual_axis_margin.py  # Revenue vs Profit Margin twinx() dual-axis
│   └── challenge5_reusable_engine.py   # Reusable create_sales_dashboard() function
├── data/
│   └── ecommerce_sales.csv         # Enterprise transactions dataset (>700 records)
├── app/
│   ├── __init__.py
│   ├── config.py                   # Central dimensions, palettes, DPI, and layout parameters
│   ├── loader.py                   # CSV loader and schema validator
│   ├── analyzer.py                 # Pure analytical KPI and aggregation calculations
│   ├── formatters.py               # Currency, percentage, and integer formatters
│   ├── charts.py                   # Individual plot functions accepting target Axes
│   ├── dashboard.py                # GridSpec-driven master dashboard generator
│   └── main.py                     # CLI orchestrator generating charts and dashboard
├── output/
│   ├── charts/
│   │   ├── monthly_revenue.png
│   │   ├── regional_revenue.png
│   │   ├── category_revenue.png
│   │   ├── top_products.png
│   │   ├── revenue_profit.png
│   │   └── quantity_distribution.png
│   └── ecommerce_dashboard.png     # Full multi-panel executive dashboard
└── tests/                          # 25+ Pytest Unit Tests (100% Passing)
    ├── __init__.py
    ├── conftest.py                 # Mock data fixtures
    ├── test_loader.py              # Ingestion unit tests
    ├── test_analyzer.py            # Analytics and KPI unit tests
    ├── test_formatters.py          # Formatter unit tests
    └── test_dashboard.py           # Chart and dashboard output tests
```

## 🚀 Running the Project
1. **Execute All Exercises**:
   ```bash
   python exercises/task1_multiple_subplots.py
   python exercises/task2_reference_line.py
   python exercises/task3_auto_annotation.py
   python exercises/task4_currency_format.py
   python exercises/task5_date_formatting.py
   python exercises/task6_gridspec_layout.py
   ```
2. **Execute All Challenges**:
   ```bash
   python coding_challenges/challenge1_target_comparison.py
   python coding_challenges/challenge2_dynamic_peak_annot.py
   python coding_challenges/challenge3_kpi_dashboard.py
   python coding_challenges/challenge4_dual_axis_margin.py
   python coding_challenges/challenge5_reusable_engine.py
   ```
3. **Generate Master Dashboard & Individual Charts**:
   ```bash
   python -m app.main
   ```
4. **Run Test Suite**:
   ```bash
   pytest
   ```
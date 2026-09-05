# Day 61 — Matplotlib Fundamentals

## 📊 Overview
Day 61 marks the transition from pure data processing and exploratory analysis into **Data Visualization** with Matplotlib. Visualizing data transforms raw statistical metrics into intuitive graphical narratives, revealing growth trajectories, category concentrations, distribution shapes, and variable associations.

## 🎯 Topics Covered
- **Data Visualization Foundations**: Principles of clarity, truthfulness, relevance, and chart selection.
- **Matplotlib Architecture**: `Figure` (the overall canvas) vs `Axes` (the coordinate plane containing ticks, lines, and titles).
- **Object-Oriented Plotting**: Using `fig, ax = plt.subplots()` for modular, multi-panel, and scalable chart generation.
- **Fundamental Chart Types**:
  - Line Charts (`ax.plot`): Time-series and trends.
  - Vertical Bar Charts (`ax.bar`): Categorical comparisons.
  - Horizontal Bar Charts (`ax.barh`): Comparisons with long category labels.
  - Scatter Plots (`ax.scatter`): Bivariate correlation & relationship audits.
  - Histograms (`ax.hist`): Frequency distribution and skewness inspection.
  - Pie Charts (`ax.pie`): Proportions of a whole (with critical analysis of when bar charts are superior).
- **Customization & Formatting**: Titles, axis labels, legends, grids, figure sizes (`figsize`), and high-resolution export (`dpi=300`, `bbox_inches="tight"`).
- **Headless & Production Best Practices**: Using non-interactive backend `matplotlib.use('Agg')` and explicit memory cleanup with `plt.close(fig)`.
- **Pandas + Matplotlib Integration**: Bridging Series/DataFrame aggregations directly into customized Matplotlib figures.

## 📂 Repository Structure
```text
Day 61/
├── Day61.md                        # Masterclass Notes, 25 Interview Q&As & Revision Test Solutions
├── pyproject.toml                  # Pytest runner configuration
├── requirements.txt                # Dependencies (pandas, numpy, matplotlib, pytest)
├── README.md                       # Comprehensive Project Documentation
├── .gitignore                      # Python/OS ignore rules
├── exercises/
│   ├── task1_line_chart.py         # Task 1: Line Chart (Monthly Revenue)
│   ├── task2_bar_chart.py          # Task 2: Bar Chart (Product Sales sorted)
│   ├── task3_horizontal_bar.py     # Task 3: Horizontal Bar Chart (Regional Revenue)
│   ├── task4_scatter_plot.py       # Task 4: Scatter Plot (Study Hours vs Marks)
│   ├── task5_histogram.py          # Task 5: Histogram (Exam Marks multi-bin comparison)
│   ├── task6_pie_chart.py          # Task 6: Pie Chart (Category Revenue Share)
│   └── task7_save_charts.py        # Task 7: Batch saving utility demonstration
├── coding_challenges/
│   ├── challenge1_sales_profit_line.py       # Challenge 1: 12-Month Sales vs Profit line
│   ├── challenge2_top5_products_bar.py       # Challenge 2: Top 5 products bar chart
│   ├── challenge3_customer_spending_hist.py   # Challenge 3: Customer spending 10-bin histogram
│   ├── challenge4_discount_profit_scatter.py  # Challenge 4: Discount vs Profit scatter + Pearson r
│   └── challenge5_revenue_rolling_avg.py     # Challenge 5: Monthly Revenue + Rolling Average overlay
├── data/
│   └── sales.csv                   # Cleaned enterprise sales transaction dataset (700+ rows)
├── app/
│   ├── __init__.py
│   ├── config.py                   # Central paths, dimensions, DPI, and styling themes
│   ├── loader.py                   # Resilient CSV data loader & validator
│   ├── analyzer.py                 # Pure analytical aggregations (GroupBy, Rolling, etc.)
│   ├── charts.py                   # Matplotlib chart renderers using Figure & Axes
│   ├── report.py                   # Formatted analytical business report generator
│   └── main.py                     # CLI pipeline orchestrator
├── output/
│   ├── charts/
│   │   ├── monthly_revenue.png
│   │   ├── regional_revenue.png
│   │   ├── category_revenue.png
│   │   ├── top_products.png
│   │   ├── top_customers.png
│   │   ├── quantity_distribution.png
│   │   ├── revenue_profit.png
│   │   ├── category_share.png
│   │   └── monthly_revenue_rolling.png
│   └── visualization_report.txt    # Executive visual intelligence report
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Pytest fixtures and mock dataset
    ├── test_loader.py              # Data loader unit tests
    ├── test_analyzer.py            # Analytical aggregations unit tests
    └── test_charts.py              # Chart generation & file output unit tests
```

## 🚀 Running the Project
1. **Run All Exercises**:
   ```bash
   python exercises/task1_line_chart.py
   python exercises/task2_bar_chart.py
   python exercises/task3_horizontal_bar.py
   python exercises/task4_scatter_plot.py
   python exercises/task5_histogram.py
   python exercises/task6_pie_chart.py
   python exercises/task7_save_charts.py
   ```
2. **Run All Coding Challenges**:
   ```bash
   python coding_challenges/challenge1_sales_profit_line.py
   python coding_challenges/challenge2_top5_products_bar.py
   python coding_challenges/challenge3_customer_spending_hist.py
   python coding_challenges/challenge4_discount_profit_scatter.py
   python coding_challenges/challenge5_revenue_rolling_avg.py
   ```
3. **Execute Main Pipeline**:
   ```bash
   python -m app.main
   ```
4. **Run Pytest Suite**:
   ```bash
   pytest
   ```

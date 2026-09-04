# Day 57 — Pandas Data Manipulation & Aggregation

Welcome to **Day 57** of the 200 Days of Python Challenge.

## 📌 Overview

Today covers advanced Pandas data manipulation:
- `groupby()` & Multi-metric Aggregations (`agg()`, Named Aggregations)
- Multi-column Grouping & `reset_index()`
- Relational Merging (`pd.merge()` inner, left, right, outer joins)
- Table Concatenation (`pd.concat()` row-wise & column-wise)
- Transformations with `apply()`, `map()`, and `replace()`
- Advanced Filtering (`isin()`, `between()`, `query()`)
- Data Reshaping with `pivot_table()`
- Time-Series Analysis (`pd.to_datetime()`, `.dt.to_period("M")`)
- **Main Project**: Sales Analytics Engine V1

## 📁 Directory Structure

```
Day 57/
├── Day57.md                  # Masterclass notes & 25 Interview Q&As
├── README.md                 # Day 57 documentation & overview
├── pyproject.toml            # Project build configuration
├── requirements.txt          # Python dependencies (pandas, pytest)
├── .gitignore                # Git ignore patterns
├── data/
│   ├── raw/
│   │   └── sales.csv         # Raw Sales Transaction Dataset (100+ rows)
│   └── processed/
│       └── cleaned_sales.csv # Cleaned & Transformed Sales Dataset
├── coding_challenges/
│   ├── challenge1_highest_dept_salary.py
│   ├── challenge2_top3_customers.py
│   ├── challenge3_highest_revenue_product_per_region.py
│   ├── challenge4_region_product_pivot.py
│   └── challenge5_region_revenue_percentage.py
├── exercises/
│   ├── task1_grouping.py
│   ├── task2_multiple_grouping.py
│   ├── task3_merge.py
│   ├── task4_concat.py
│   ├── task5_apply.py
│   ├── task6_map.py
│   └── task7_pivot_table.py
├── app/
│   ├── __init__.py
│   ├── loader.py             # Data loading & schema validation
│   ├── cleaner.py            # Data cleaning pipeline
│   ├── transformer.py        # Revenue & Month period calculation
│   ├── analyzer.py           # Business analytics & pivot tables engine
│   ├── report.py             # Executive ASCII report generator & exporter
│   └── main.py               # CLI entry point
├── output/
│   ├── sales_report.txt      # Formatted Executive ASCII Report
│   └── regional_category_summary.csv # Regional x Category Pivot Export
└── tests/
    ├── conftest.py           # Test fixtures
    ├── test_loader.py        # Tests for loader module
    ├── test_cleaner.py       # Tests for cleaner module
    ├── test_transformer.py   # Tests for transformer module
    └── test_analyzer.py      # Tests for analyzer module
```

## 🚀 How to Run

### Run CLI Application
```bash
python "Day 57/app/main.py"
```

### Run Automated Unit Tests
```bash
pytest "Day 57/tests/"
```

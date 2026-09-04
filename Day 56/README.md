# Day 56 — Pandas Fundamentals: Series & DataFrames

Welcome to **Day 56** of the 200 Days of Python Challenge.

## 📁 Directory Structure

```
Day 56/
├── Day56.md                  # Masterclass notes & 25 Interview Q&As
├── README.md                 # Day 56 documentation & overview
├── pyproject.toml            # Project build configuration
├── requirements.txt          # Python dependencies (pandas, pytest)
├── .gitignore                # Git ignore patterns
├── data/
│   ├── raw/
│   │   └── students.csv      # Raw student dataset
│   └── processed/
│       └── student_analysis.csv # Cleaned & computed output dataset
├── coding_challenges/
│   ├── challenge1_passing_students.py
│   ├── challenge2_top2_students.py
│   ├── challenge3_grade_column.py
│   ├── challenge4_average_marks.py
│   └── challenge5_product_revenue.py
├── exercises/
│   ├── task1_series_basics.py
│   ├── task2_dataframe_analytics.py
│   ├── task3_filtering_practice.py
│   ├── task4_sorting_practice.py
│   └── task5_calculated_columns.py
├── app/
│   ├── __init__.py
│   ├── loader.py             # Data loading module
│   ├── cleaner.py            # Data cleaning pipeline
│   ├── analyzer.py           # Analysis & metrics calculator
│   ├── report.py             # ASCII report generator & exporter
│   └── main.py               # Interactive CLI application
└── tests/
    ├── conftest.py           # Test fixtures
    ├── test_loader.py        # Tests for loader module
    ├── test_cleaner.py       # Tests for cleaner module
    └── test_analyzer.py      # Tests for analyzer module
```

## 🚀 How to Run

### Run CLI Application
```bash
python -m Day_56.app.main
# or
python "Day 56/app/main.py"
```

### Run Automated Unit Tests
```bash
pytest "Day 56/tests/"
```

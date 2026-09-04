# Day 58 — Pandas Data Cleaning & Preprocessing

Welcome to **Day 58** of the 200 Days of Python Challenge.

## 📌 Topics Covered

- Missing values & `NaN` (`isna()`, `isnull()`, `dropna()`, `fillna()`, `ffill()`, `bfill()`)
- Mean vs Median imputation strategies
- Duplicate detection & removal (`duplicated()`, `drop_duplicates()`)
- String normalization (`str.strip()`, `str.lower()`, `str.title()`, `str.replace()`, `str.contains()`)
- Data type coercion (`astype()`, `pd.to_numeric(errors="coerce")`)
- Date conversion (`pd.to_datetime(errors="coerce")`, `.dt.year`, `.dt.month`, `.dt.month_name()`)
- Domain validation & range filtering (`between()`)
- Monetary string parsing (`₹`, `$`, `,`)
- Category standardization (Gender: `Male`/`Female`/`Unknown`)
- **Main Project**: Real-World Messy Dataset Cleaning Pipeline

## 📁 Directory Structure

```
Day 58/
├── Day58.md                  # Masterclass notes & 30 Interview Q&As
├── README.md                 # Day 58 documentation & overview
├── pyproject.toml            # Project build configuration
├── requirements.txt          # Python dependencies (pandas, numpy, pytest)
├── .gitignore                # Git ignore patterns
├── data/
│   ├── raw/
│   │   └── messy_customers.csv # Raw Messy Customer Dataset (120+ rows)
│   └── processed/
│       └── clean_customers.csv # Cleaned & Validated Customer Dataset
├── coding_challenges/
│   ├── challenge1_missing_value_statistics.py
│   ├── challenge2_duplicate_customer_detection.py
│   ├── challenge3_monetary_conversion.py
│   ├── challenge4_category_standardization.py
│   └── challenge5_reusable_cleaner_pipeline.py
├── exercises/
│   ├── task1_missing_values.py
│   ├── task2_duplicates.py
│   ├── task3_string_cleaning.py
│   ├── task4_numeric_cleaning.py
│   ├── task5_date_cleaning.py
│   └── task6_validation.py
├── app/
│   ├── __init__.py
│   ├── loader.py             # Data ingestion & schema inspection
│   ├── cleaner.py            # Deduplication, string/numeric/date cleaning & imputation
│   ├── validator.py          # Data rules validation engine
│   ├── analyzer.py           # Data quality metrics & before/after statistics
│   ├── report.py             # Data quality ASCII report generator & exporter
│   └── main.py               # CLI entry point
├── output/
│   └── data_quality_report.txt # Generated Executive Data Quality Report
└── tests/
    ├── conftest.py           # Test fixtures & sample datasets
    ├── test_loader.py        # Tests for loader module
    ├── test_cleaner.py       # Tests for cleaner module
    ├── test_validator.py     # Tests for validator module
    └── test_analyzer.py      # Tests for analyzer module
```

## 🚀 How to Run

### Run CLI Application
```bash
python "Day 58/app/main.py"
```

### Run Automated Unit Tests
```bash
pytest "Day 58/tests/"
```

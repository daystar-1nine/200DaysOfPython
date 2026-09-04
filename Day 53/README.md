# Day 53 — Data Processing with Python

Welcome to **Day 53** of the **200 Days of Python Challenge**! Today's focus is on mastering **Real-World Data Processing, Cleaning, Transformation & Analysis in Python**, establishing the data pipeline foundation necessary before transitioning into NumPy and Pandas.

---

## 🎯 Topics Covered

- **Data Cleaning**: String stripping, title casing, safe type parsing (`safe_int`, `safe_float`, `safe_date`), handling missing values, and record normalization.
- **Data Validation**: Domain boundary checks for order ID, price, quantity, date formats, and string completeness.
- **Data Transformation & Deduplication**: Computing derived fields (`total = price * quantity`), mapping dicts to `@dataclass` models, and deduplicating by primary key.
- **Data Analysis & Aggregations**: Financial summaries, mean AOV, extremal order identification (`max(key=...)`), category grouping (`defaultdict`), product counting (`Counter`), and ranking.
- **In-Memory Relational Joins**: Replicating SQL inner joins and data warehouse star schemas using pure Python dictionary hash maps.
- **Data Pipelines**: Modular separation of concerns (Ingest -> Inspect -> Clean -> Validate -> Deduplicate -> Transform -> Analyze -> Report -> Export).

---

## 📂 Project Architecture

```text
Day 53/
├── Day53.md                        # Masterclass theory notes & 20 interview Q&As
├── README.md                       # Project documentation (this file)
├── pyproject.toml                  # Pytest & tool configurations
├── requirements.txt                # Dependencies (pytest)
├── app/                            # Sales Data Processing System package
│   ├── __init__.py
│   ├── models.py                   # Dataclass Sale entity & derived total property
│   ├── csv_loader.py               # Raw CSV dataset loader with schema validation
│   ├── cleaner.py                  # Normalization & safe numeric/date parsing helpers
│   ├── validator.py                # Business domain boundary validation
│   ├── transformer.py              # Transformation & order_id deduplication engine
│   ├── analyzer.py                 # Revenue, AOV, top products & category analytics
│   ├── reporter.py                 # ASCII summary report generator & CSV exporter
│   └── main.py                     # Pipeline execution & interactive CLI entry point
├── coding_challenges/              # 5 Standalone coding challenges
│   ├── challenge1_clean_names.py
│   ├── challenge2_remove_duplicates.py
│   ├── challenge3_category_revenue.py
│   ├── challenge4_top_product.py
│   └── challenge5_sales_pipeline.py
├── exercises/                      # Mini Data Engineering exercises
│   ├── exercise1_customer_join.py  # Python inner join (sales + customers)
│   └── exercise2_multi_join_warehouse.py # 3-way data warehouse join (products + sales + customers)
├── data/                           # Datasets
│   ├── raw/
│   │   ├── sales.csv               # Raw dataset with whitespace, duplicates & bad rows
│   │   ├── customers.csv           # Customer dimension metadata
│   │   └── products.csv            # Product dimension metadata
│   └── processed/
│       └── cleaned_sales.csv       # Output cleaned dataset
├── output/
│   └── sales_report.txt            # Generated ASCII executive analysis report
└── tests/                          # Automated Pytest suite (24 passing tests)
    ├── conftest.py                 # Test fixtures & temp files
    ├── test_cleaner.py             # String cleaning & safe type casting tests
    ├── test_validator.py           # Domain boundary validation tests
    ├── test_csv_loader.py          # CSV loader unit tests
    ├── test_transformer.py        # Model transformation & deduplication unit tests
    ├── test_analyzer.py           # Financial metrics & ranking unit tests
    └── test_reporter.py           # CSV export & report generator unit tests
```

---

## 🔄 Sales Data Pipeline Workflow

```text
sales.csv (Raw Dataset)
        │
        ↓
   csv_loader.py (Load Raw Records)
        │
        ↓
    cleaner.py (Normalize Strings & Safe Numeric/Date Parsing)
        │
        ↓
   validator.py (Validate Price, Quantity & ID Boundaries)
        │
        ↓
 transformer.py (Deduplicate Order IDs & Instantiating Sale Models)
        │
        ↓
   analyzer.py (Compute Revenue, AOV, Top Products & Category Aggregations)
        │
        ↓
   reporter.py (Export cleaned_sales.csv & Generate sales_report.txt)
```

---

## 🧪 Testing & Verification

Run the full automated test suite using `pytest`:

```bash
pytest "Day 53/tests/"
```

All 24 test cases verify string normalization, safe type casting, domain boundary validation, deduplication, derived calculations, aggregations, and reporting.

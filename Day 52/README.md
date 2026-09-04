# Day 52 — JSON, CSV & Data Serialization in Python

Welcome to **Day 52** of the **200 Days of Python Challenge**! Today's focus is on mastering **JSON, CSV & Data Serialization in Python**, building a bridge between pythonic object models and persistent structured file formats.

---

## 🎯 Day 52 Goals

By completing Day 52, you have mastered:
- **Serialization & Deserialization**: Converting Python dataclasses and nested dictionary structures into JSON/CSV data streams and vice-versa.
- **Python Standard Library IO**: Utilizing `json` (`dumps()`, `loads()`, `dump()`, `load()`) and `csv` (`DictWriter`, `DictReader`) with context managers and UTF-8 encoding.
- **Dataclass Integration**: Leveraging `@dataclass`, `asdict()`, `**kwargs` unpacking, and custom instance factory methods.
- **Defensive Error Handling**: Catching `json.JSONDecodeError`, `FileNotFoundError`, `KeyError`, and `ValueError` to handle corrupt or malformed files gracefully.
- **Data Validation Pipelines**: Validating entity attributes (ID, name, age, course, marks) before persistence.
- **Data Analytics & Reports**: Calculating grade statistics, highest/lowest scores, and department distributions using `collections.Counter`.

---

## 📂 Project Architecture

```text
Day 52/
├── Day52.md                        # Masterclass theory notes & 20 interview Q&As
├── README.md                       # Project documentation (this file)
├── pyproject.toml                  # Pytest & tool configurations
├── requirements.txt                # Dependencies (pytest)
├── app/                            # Student Data Management System V2 package
│   ├── __init__.py
│   ├── models.py                   # Dataclass entity definition
│   ├── validators.py               # Field validation rules
│   ├── json_handler.py             # JSON file load & save handler
│   ├── csv_handler.py              # CSV file load & save handler
│   ├── services.py                 # CRUD & statistical business logic
│   ├── reports.py                  # Grade statistics & ASCII report generator
│   ├── utils.py                    # Menu helpers & input prompt utilities
│   └── main.py                     # Interactive 14-option CLI entry point
├── coding_challenges/              # 5 Standalone coding challenges
│   ├── challenge1_json_serialization.py
│   ├── challenge2_json_highest_score.py
│   ├── challenge3_csv_average.py
│   ├── challenge4_format_conversion.py
│   └── challenge5_data_cleaning.py
├── exercises/                      # Hands-on practice exercises
│   └── practice1_university_json.py# Nested university structure serializer
├── data/                           # Seed datasets
│   ├── students.json
│   └── students.csv
└── tests/                          # Automated Pytest suite (27 passing tests)
    ├── conftest.py                 # Test fixtures
    ├── test_validators.py          # Field validation tests
    ├── test_json_handler.py        # JSON serialization unit tests
    ├── test_csv_handler.py         # CSV serialization unit tests
    └── test_services.py            # Business logic unit tests
```

---

## 🚀 Interactive CLI Options (App V2)

The interactive CLI (`python app/main.py`) provides 14 management capabilities:

1. **List All Students** — Display formatted table of students.
2. **Add New Student** — Prompt inputs, validate, and assign unique incrementing ID.
3. **Update Student** — Modify specific student fields by ID.
4. **Delete Student** — Remove student record by ID.
5. **Search Students** — Filter roster by ID, name substring, or course.
6. **Show Grade Statistics** — View mean score, highest score, lowest score, and distribution.
7. **Generate ASCII Analytical Report** — Print formatted report with `Counter` distributions.
8. **Export Data to JSON** — Save current roster to `data/students.json`.
9. **Import Data from JSON** — Load and validate roster from JSON file.
10. **Export Data to CSV** — Save current roster to `data/students.csv`.
11. **Import Data from CSV** — Load and validate roster from CSV file.
12. **Convert JSON to CSV** — Direct file format transformation.
13. **Convert CSV to JSON** — Direct file format transformation.
14. **Exit** — Save state and exit application cleanly.

---

## 🧪 Testing & Verification

Run the full automated test suite using `pytest`:

```bash
pytest "Day 52/tests/"
```

All 27 test cases test model validation, serialization, deserialization, exception catching, and business logic.

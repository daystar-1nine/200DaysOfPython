# 📊 Day 51 — Professional Python Student Data Processor CLI

> A clean, type-annotated, modular command-line application built using modern Python standard libraries (**Type Hints**, **Dataclasses**, **Enums**, **Pathlib**, **Datetime**, and **Collections**).

---

## 📌 Topics Covered
- **Type Hints**: Explicit function signatures, generics (`list[T]`, `dict[K, V]`), `Union` (`T | None`), and `TypeAlias`.
- **Dataclasses**: `@dataclass` with `@property` dynamics and `field(default_factory=...)`.
- **Enums**: `PerformanceLevel(str, Enum)` state representation.
- **Pathlib**: `Path.mkdir()`, `Path.read_text()`, `Path.write_text()`, and path joining (`/`).
- **Datetime**: ISO UTC timestamp formatting (`strftime`).
- **Collections**: `collections.Counter` frequency tallies, `defaultdict`, and `deque`.
- **Clean Code & Architecture**: PEP 8 style, Single Responsibility Principle, small functions.
- **Automated Testing**: 19 Pytest test cases covering CSV parsing, edge cases, and report generation.

---

## 📁 Repository Structure

```text
Day 51/
├── Day51.md                   # Masterclass Notes & 15 Technical Interview Answers
├── coding_challenges/
│   ├── challenge1_counter.py
│   ├── challenge2_highest_student.py
│   ├── challenge3_pathlib.py
│   ├── challenge4_employee_dataclass.py
│   └── challenge5_customer_queue.py
├── app/
│   ├── __init__.py
│   ├── enums.py               # PerformanceLevel Enum & get_performance_level
│   ├── models.py              # Student Dataclass
│   ├── file_handler.py        # Pathlib & CSV parser
│   ├── services.py            # Student statistics & Counter distribution
│   ├── reports.py             # ASCII report generator & Pathlib file write
│   ├── utils.py               # Table formatter
│   └── main.py                # Interactive CLI Application
├── data/
│   └── raw/
│       └── students.csv       # Raw CSV input dataset
├── output/
│   └── report.txt             # Generated output report
├── tests/
│   ├── conftest.py            # Test data & tmp_path fixtures
│   ├── test_services.py       # Analytics unit tests
│   ├── test_file_handler.py   # CSV parsing & error validation tests
│   └── test_reports.py        # Report generation tests
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Running Automated Pytest Suite

```bash
# Execute Pytest test suite (19 passing test cases)
pytest "Day 51/tests/" -v
```

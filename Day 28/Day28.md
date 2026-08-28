# 🐍 Day 28/200 – Masterclass Notes: Python Project Architecture & Packaging

🎯 **Goal:** Learn how to structure Python projects like professional software engineers—modules vs packages, `__init__.py`, imports (absolute & relative), virtual environments (`.venv`), `pyproject.toml`, `src/` layout architecture, **Separation of Concerns** (CLI $\rightarrow$ Service $\rightarrow$ Database $\rightarrow$ Models), editable installation (`pip install -e .`), entry points (`__main__.py`), and **Dependency Injection**.

---

## 📌 Executive Summary & Key Takeaways

- **Module vs. Package:**
  - **Module:** A single `.py` file containing functions, classes, or constants (e.g. `math_utils.py`).
  - **Package:** A directory containing an `__init__.py` file and multiple related Python modules.
- **The `src/` Layout Architecture:** Placing application packages inside a `src/` directory (e.g. `src/expense_tracker/`) prevents accidental imports of uninstalled development code and enforces clean packaging testing.
- **Modern Packaging with `pyproject.toml`:** The unified standard (PEP 517/518/621) for project metadata, dependencies, and CLI executable entry points (`[project.scripts]`).
- **Separation of Concerns (Layered Architecture):**
  1. **CLI / Presentation Layer (`cli/`):** Handles `argparse` user command parsing and output formatting.
  2. **Service Layer (`services/`):** Encapsulates business validation, calculations, and rules.
  3. **Database Layer (`database.py`):** Handles raw SQL queries, connections, and transactions.
  4. **Domain Models (`models/`):** `@dataclass` containers representing domain entities (`Expense`, `User`).
- **Dependency Injection:** Constructing services by passing database dependencies into their `__init__(database)` constructors, enabling effortless swapping between production, staging, and mock test databases.

---

## 📖 Topic 1: Modules, Packages & Entry Points

### 1.1 `__init__.py` & Module Execution (`__main__.py`)

```python
# src/expense_tracker/__init__.py
"""Package initialization and top-level exports."""
__version__ = "0.1.0"

# src/expense_tracker/__main__.py
"""Allows running package directly via `python -m expense_tracker`."""
from .main import main

if __name__ == "__main__":
    main()
```

---

## 📖 Topic 2: Modern `pyproject.toml` Packaging

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "expense-tracker"
version = "0.1.0"
description = "A professional CLI multi-user expense tracker"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "requests>=2.28.0",
    "python-dotenv>=1.0.0"
]

[project.scripts]
expense-tracker = "expense_tracker.main:main"
```

---

## 📖 Topic 3: Layered Architecture & Dependency Injection

```python
# 1. Database Layer (Isolated SQL)
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def save_expense(self, user_id: int, category_id: int, amount: float) -> int:
        # SQL logic here
        return 1

# 2. Service Layer with Dependency Injection
class ExpenseService:
    def __init__(self, database: Database):  # Database injected via constructor
        self.database = database

    def add_expense(self, user_id: int, category_id: int, amount: float) -> int:
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        return self.database.save_expense(user_id, category_id, amount)

# 3. Application Wiring (main.py)
db = Database("expenses.db")
service = ExpenseService(db)  # Dependency Injection in action
```

---

## ⚡ Master Cheat Sheet

```python
# Project Architecture Master Cheat Sheet

# 1. Editable Package Installation
# Terminal: pip install -e .

# 2. Running Package as a Module
# Terminal: python -m expense_tracker

# 3. Layered Import Architecture (src/expense_tracker/)
from expense_tracker.models.expense import Expense
from expense_tracker.services.expense_service import ExpenseService
from expense_tracker.database import Database
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Coupling Business Logic Directly to CLI or SQL Queries:**
   - ❌ Putting `sqlite3.connect()` calls directly inside `argparse` command handlers.
   - ✅ Delegate CLI requests to `ExpenseService`, which delegates queries to `Database`.

2. **Hardcoding Database Paths inside Service Classes:**
   - ❌ `class ExpenseService: def __init__(self): self.db = sqlite3.connect("expenses.db")`.
   - ✅ Pass the `Database` instance into `ExpenseService.__init__(self, database)` via Dependency Injection.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What does `pip install -e .` do?
**Answer:** It installs the package in "editable" mode. Setuptools creates a link pointing directly to your local `src/` development code, allowing changes made to source files to take effect immediately without needing to reinstall the package after every edit.

### Q2: What is the benefit of Dependency Injection in Python application architecture?
**Answer:** Dependency Injection decouples high-level business services from low-level infrastructure implementations (like database drivers or external APIs). It makes application code modular, flexible, and effortlessly testable by allowing developers to pass mock or in-memory test databases into services during automated testing.

---

## 📝 Recap Checklist
- [x] Differentiated between Python modules and packages.
- [x] Structured projects using standard `src/` layout architecture.
- [x] Configured `pyproject.toml` metadata and CLI scripts (`[project.scripts]`).
- [x] Implemented Separation of Concerns (CLI $\rightarrow$ Service $\rightarrow$ Database $\rightarrow$ Models).
- [x] Applied Dependency Injection pattern across service constructors.
- [x] Configured runnable package entry points (`__main__.py`).
- [x] Built fully refactored, packageable Multi-User Expense Tracker project with test suite.

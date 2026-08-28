# 🐍 Day 30/200 – Python Foundation Capstone: PyFinance CLI

🎯 **Capstone Mission:** Build **PyFinance**, a production-style, multi-layer Personal Finance Manager CLI application combining all Foundation skills—Python OOP, Dataclasses, Type Hints, Relational Database Design (SQLite), Repository Pattern, Service Layer with **Dependency Injection**, REST API Integration with Session reuse & retries, Centralized Configuration (`.env`), Custom Exception Hierarchy, Modular Logging, Professional CLI formatting, and a Pytest suite (Unit, Integration, and API Mocking tests).

---

## 📌 Executive Summary & Architecture Overview

- **Layered DDD & Repository Pattern Architecture:**
  - **Presentation / CLI Layer (`cli/`):** Styled terminal subcommand handlers (`add`, `list`, `update`, `delete`, `search`, `report`, `currency`, `budget`, `export`, `import`).
  - **Service Layer (`services/`):** Business logic validation (`ExpenseService`, `ReportService`, `CurrencyService`, `BudgetService`).
  - **Repository Layer (`repositories/`):** Data access abstraction (`ExpenseRepository`) shielding business logic from SQL specifics.
  - **Database Layer (`database.py`):** SQLite connection management with foreign key enforcement (`PRAGMA foreign_keys = ON;`).
  - **External API Client Layer (`api/`):** Currency exchange rate client using `requests.Session()`, retries, timeouts, and local JSON caching.
  - **Domain Models (`models/`):** Type-safe `@dataclass` objects (`Expense`, `Budget`, `CurrencyRate`).

```text
                             USER TERMINAL
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │                  CLI Layer                          │ (cli/commands.py)
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │                Service Layer                        │ (services/expense_service.py)
        │      (Business Validation & Calculations)            │ (services/report_service.py)
        └─────────────┬─────────────────────────┬─────────────┘ (services/currency_service.py)
                      │                         │
                      ▼                         ▼
        ┌───────────────────────────┐ ┌───────────────────┐
        │    Repository Layer       │ │  API Client Layer │ (api/client.py)
        │  (ExpenseRepository)      │ └─────────┬─────────┘
        └─────────────┬─────────────┘           │
                      │                         ▼
                      ▼                   EXTERNAL API
        ┌───────────────────────────┐  (Exchange Rates)
        │     SQLite Database       │
        └───────────────────────────┘
```

---

## 📖 Topic 1: Database Schema & Entity Design

```sql
-- SQLite Database Schema (data/pyfinance.db)

CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL NOT NULL CHECK(amount > 0),
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    date TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budgets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT UNIQUE NOT NULL,
    monthly_limit REAL NOT NULL CHECK(monthly_limit > 0)
);

CREATE INDEX IF NOT EXISTS idx_expenses_category_date ON expenses(category, date);
```

---

## 📖 Topic 2: Repository Abstraction Pattern

```python
# src/pyfinance/repositories/expense_repository.py
import sqlite3
from pyfinance.models.expense import Expense

class ExpenseRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def add(self, expense: Expense) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, description, date, created_at) VALUES (?, ?, ?, ?, ?)",
                (expense.amount, expense.category, expense.description, expense.date, expense.created_at)
            )
            conn.commit()
            return cursor.lastrowid
```

---

## 📖 Topic 3: Service Layer & Dependency Injection

```python
# src/pyfinance/services/expense_service.py
from pyfinance.repositories.expense_repository import ExpenseRepository
from pyfinance.exceptions import ValidationError, NotFoundError

class ExpenseService:
    def __init__(self, repository: ExpenseRepository):
        # Dependency Injection via constructor
        self.repository = repository

    def create_expense(self, amount: float, category: str, description: str, date: str) -> int:
        if amount <= 0:
            raise ValidationError("Amount must be greater than zero.")
        if not category.strip():
            raise ValidationError("Category cannot be empty.")
        # Delegate storage to repository
        return self.repository.add(Expense(None, amount, category, description, date, "..."))
```

---

## ⚡ Master CLI Command Reference

```bash
# Terminal Usage Reference

# 1. Expense Management
pyfinance add --amount 250 --category Food --description "Lunch at cafe"
pyfinance list
pyfinance search --category Food
pyfinance search --from 2026-08-01 --to 2026-08-30 --keyword Lunch
pyfinance update 1 --amount 300
pyfinance delete 1

# 2. Financial Analytics Reports
pyfinance report total
pyfinance report category
pyfinance report monthly

# 3. Currency Conversion API
pyfinance currency USD INR

# 4. Budget & Export Management
pyfinance budget set --category Food --limit 5000
pyfinance budget status
pyfinance export --format csv --output expenses.csv
pyfinance import --format csv --input expenses.csv
```

---

## 📝 Definition of Done Checklist
- [x] SQLite database schema with indexes and transaction safety.
- [x] Full CRUD operations (Create, Read, Update, Delete).
- [x] Repository pattern isolating SQL queries from business logic.
- [x] Service Layer implementing business validation & aggregations.
- [x] REST API Currency Service with session reuse, retries, and local cache.
- [x] CLI presentation layer with styled ASCII headers and tabular formatting.
- [x] Centralized `.env` configuration & custom exception hierarchy.
- [x] Comprehensive Pytest suite (Unit, Integration, and Mock tests).
- [x] Modern `pyproject.toml` package configuration (`pip install -e .`).

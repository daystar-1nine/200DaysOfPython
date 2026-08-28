# Professional Refactored Expense Tracker CLI Package

A production-grade Python CLI application structured using modern `src/` layout architecture, `pyproject.toml`, layered separation of concerns, and **Dependency Injection**.

## Architecture & Layering

```text
               USER (CLI)
                   │
                   ▼
        ┌─────────────────────┐
        │  CLI Layer          │ (cli/commands.py)
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Service Layer      │ (services/expense_service.py)
        │  Business Logic     │ (services/report_service.py)
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Database Layer     │ (database.py)
        └─────────────────────┘
```

## Installation & Usage

```bash
# 1. Editable Installation
pip install -e .

# 2. Command Line Executable Usage
expense-tracker add --category Food --amount 250
expense-tracker list
expense-tracker summary
expense-tracker report --month 08 --year 2026

# 3. Module Execution Mode
python -m expense_tracker
```

# PyFinance — Personal Finance Manager CLI 🚀

PyFinance is a production-style, multi-layer command-line application for personal finance management built with Python, SQLite, REST APIs, and Pytest.

## Features

- 💸 **Expense Management:** Full CRUD operations (Add, List, Update, Delete).
- 🔍 **Advanced Search:** Search by Category, Date Range (`--from`, `--to`), or Keyword.
- 📊 **Financial Analytics:** Total Spending, Category Breakdown, and Monthly Trend Reports.
- 🌐 **Live Currency Conversion:** Real-time exchange rate updates powered by Currency API with local response caching.
- 💰 **Category Budget Tracking:** Set monthly category budgets and track spending limits.
- 📁 **Data Export & Import:** Export/Import expenses to CSV and JSON formats.
- 🛡️ **Robust Architecture:** Layered Domain-Driven Design, Repository Pattern, Dependency Injection, Centralized Config, and Pytest Test Suite.

## Installation

```bash
# 1. Clone repository & install in editable mode
pip install -e .

# 2. Verify installation
pyfinance --help
```

## CLI Usage Examples

```bash
# Add Expense
pyfinance add --amount 250 --category Food --description "Lunch at Cafe"

# List Expenses
pyfinance list

# Search Expenses
pyfinance search --category Food --keyword Lunch

# Generate Reports
pyfinance report total
pyfinance report category
pyfinance report monthly

# Currency Conversion
pyfinance currency USD INR

# Budget Tracking
pyfinance budget set --category Food --limit 5000
pyfinance budget status

# Export / Import
pyfinance export --format csv --output expenses.csv
pyfinance import --format csv --input expenses.csv
```

## Testing

```bash
# Run complete Pytest test suite
pytest
```

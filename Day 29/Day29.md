# 🐍 Day 29/200 – Masterclass Notes: Professional Testing with Pytest

🎯 **Goal:** Master professional Python software testing using `pytest`—writing clean test functions, native `assert` statements, parameterized testing (`@pytest.mark.parametrize`), testing exception handling (`pytest.raises`), reusable data fixtures (`@pytest.fixture`, `conftest.py`), Unit vs. Integration testing, Mocking HTTP APIs (`unittest.mock.patch`), testing SQLite database persistence, CLI command testing, and measuring test coverage (`pytest-cov`).

---

## 📌 Executive Summary & Key Takeaways

- **Why Automated Testing Matters:** Manual testing after every code change is slow, error-prone, and unscalable. Automated tests guarantee code correctness, prevent regressions, and make refactoring safe.
- **Pytest Discovery & Conventions:** Pytest automatically discovers files matching `test_*.py` or `*_test.py` and executes functions prefixed with `test_*()`.
- **Pytest Fixtures (`@pytest.fixture`):** Modular, reusable setup and teardown helpers. Shared fixtures declared in `conftest.py` are automatically available to all test files in the directory tree without explicit imports.
- **Parameterized Testing:** `@pytest.mark.parametrize("input,expected", [...])` runs a single test function across multiple data tuples, generating distinct test reports for each case.
- **Mocking External Dependencies (`unittest.mock`):** Use `@patch` to replace external API calls (`requests.get`) or database connections with mock objects (`Mock()`), isolating tests from network failures and API rate limits.
- **Unit vs. Integration Tests:**
  - **Unit Tests:** Verify individual pure functions or classes in isolation.
  - **Integration Tests:** Verify multiple integrated system layers (CLI $\rightarrow$ Service $\rightarrow$ SQLite Database).

---

## 📖 Topic 1: Pytest Basics, Parameterization & Exception Testing

```python
import pytest

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# 1. Parameterized Test
@pytest.mark.parametrize("a, b, expected", [
    (10.0, 2.0, 5.0),
    (9.0, 3.0, 3.0),
    (-6.0, 2.0, -3.0),
    (0.0, 5.0, 0.0)
])
def test_divide_valid(a: float, b: float, expected: float) -> None:
    assert divide(a, b) == expected

# 2. Testing Exception and Match Message
def test_divide_zero_raises_exception() -> None:
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10.0, 0.0)
```

---

## 📖 Topic 2: Pytest Fixtures & Shared `conftest.py`

```python
# conftest.py
import pytest, os, sqlite3

@pytest.fixture
def temp_db(tmp_path):
    """Provides isolated temporary SQLite database fixture."""
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
    conn.commit()
    conn.close()
    
    yield str(db_file)  # Test runs here
    
    if os.path.exists(db_file):
        os.remove(db_file)

# test_db.py
def test_user_insert(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.execute("INSERT INTO users (name) VALUES ('Suraj');")
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users;")
    assert cursor.fetchone()[0] == "Suraj"
    conn.close()
```

---

## 📖 Topic 3: Mocking API Calls (`unittest.mock.patch`)

```python
from unittest.mock import patch, Mock
import requests

def fetch_user_github_stars(username: str) -> int:
    resp = requests.get(f"https://api.github.com/users/{username}", timeout=5)
    resp.raise_for_status()
    return resp.json().get("public_repos", 0)

@patch("requests.get")
def test_fetch_user_github_stars(mock_get):
    # Configure mock return object
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"public_repos": 42}
    mock_get.return_value = mock_resp

    result = fetch_user_github_stars("octocat")
    assert result == 42
    mock_get.assert_called_once_with("https://api.github.com/users/octocat", timeout=5)
```

---

## ⚡ Master Cheat Sheet

```python
# Pytest Master Cheat Sheet

# 1. Terminal Execution Commands
# pytest -v                          # Run all tests verbosely
# pytest tests/unit/                 # Run specific folder
# pytest -k "test_add"               # Run matching test names
# pytest --cov=src                   # Run with code coverage report

# 2. Key Fixture Scope Options
# @pytest.fixture(scope="function")  # Default: runs per test function
# @pytest.fixture(scope="module")    # Runs once per test file module
# @pytest.fixture(scope="session")   # Runs once per entire test suite run
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Mocking at the Import Origin Instead of Target Lookup Site:**
   - ❌ `@patch("requests.get")` inside a module that imported `from requests import get`.
   - ✅ `@patch("target_module.get")` to target where the symbol is looked up during execution.

2. **Modifying Real Production Databases During Tests:**
   - ❌ Running unit tests against `expenses.db`.
   - ✅ Use pytest `tmp_path` fixture or `:memory:` databases to ensure absolute test isolation.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the purpose of `conftest.py` in Pytest?
**Answer:** `conftest.py` is a special root file used by Pytest to define shared fixtures, hooks, and plugins. Any fixtures declared in `conftest.py` are automatically discovered and accessible by all test files within that directory hierarchy without needing explicit imports.

### Q2: What is the difference between `@patch` and `Mock()` in Python testing?
**Answer:** `Mock()` instantiates a fake object whose attributes, return values, and methods can be pre-configured. `@patch` is a decorator/context manager that temporarily replaces a real attribute or module in the application namespace with a `Mock` object for the duration of the test.

---

## 📝 Recap Checklist
- [x] Installed `pytest` and `pytest-cov`.
- [x] Wrote unit tests using native `assert` statements.
- [x] Created parameterized tests with `@pytest.mark.parametrize`.
- [x] Tested exception handling and error messages with `pytest.raises`.
- [x] Built reusable test data fixtures using `@pytest.fixture` and `conftest.py`.
- [x] Mocked HTTP API responses (`200`, `401`, `404`, `429`, `500`, Timeout) with `@patch`.
- [x] Built Unit & Integration test suites for Expense Tracker & GitHub API Client.

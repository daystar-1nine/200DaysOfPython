# Audit Bugs — Days 1–100

## Critical
**Problem**: Exposed Secrets and Hardcoded Credentials
**Files**:
- `Day 10/secure_login_system.py`
- `Day 47/tests/test_auth.py`
- `Day 48/tests/test_security.py`
- `Day 49/tests/test_security.py`
**Why it matters**: Storing passwords or JWT secrets directly in source control is a massive security vulnerability and bad practice.
**Suggested fix**: Extract secrets into `.env` files and use `python-dotenv` or `os.environ` to read them. Update `.gitignore` to ensure `.env` is never committed.

## High
**Problem**: Fake Test Execution (False Positives)
**Files**: Numerous `test_*.py` files in the 150 empty files list or containing `pass`.
**Why it matters**: A test file that is empty or just says `pass` will be marked as "passed" or "ignored" by Pytest, inflating test execution metrics without actually verifying functionality.
**Suggested fix**: Delete empty test files or implement `assert` statements that verify the core logic.

## Medium
**Problem**: `TODO` and `pass` in Core Logic
**Files**: `Day 50/alembic/versions/001_initial_schema.py`, Day 82-87 `coding_challenges`.
**Why it matters**: Claiming a day is complete when the core architectural code is literally a `pass` statement fundamentally violates the learning objective.
**Suggested fix**: Complete the implementation or mark the day as `INCOMPLETE` in the tracker.

## Low
**Problem**: Residual Boilerplate
**Files**: 150 `0 byte` files scattered across Day 1-81.
**Why it matters**: Clutters the repository, confuses navigation, and makes the project look artificially large.
**Suggested fix**: Run a cleanup script to aggressively purge 0-byte `.py` files that are not `__init__.py`.

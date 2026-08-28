# 🐍 Day 37/200 – Masterclass Notes: Advanced Exception Handling & Error Systems

🎯 **Goal:** Design a production-grade **Application Error System** in Python—mastering custom exception hierarchies (`ApplicationError` base class), explicit exception chaining (`raise ... from err`), preserving original cause tracebacks, catch-all error boundaries, and mapping low-level database or HTTP errors to clean domain exceptions (`ValidationError`, `DatabaseError`, `NotFoundError`, `AuthenticationError`, `ExternalServiceError`).

---

## 📌 Executive Summary & Key Takeaways

- **Application Error Hierarchy:** Inheriting from a common root base class (`ApplicationError(Exception)`) allows top-level CLI error handlers or API gateways to catch all application-specific errors with a single `except ApplicationError:` handler.
- **Explicit Exception Chaining (`raise ... from err`):** Re-raises low-level runtime errors (e.g. `sqlite3.IntegrityError` or `requests.HTTPError`) as high-level domain exceptions (`DatabaseError` or `ExternalServiceError`) while preserving the original cause in `err.__cause__`.
- **Error Boundaries:** A outer protective boundary (usually in presentation CLI or API controllers) that intercepts uncaught `ApplicationError` instances, formatting user-friendly warning messages without leaking raw stack trace details to users.

---

## 📖 Topic 1: Application Exception Hierarchy

```python
class ApplicationError(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code

class ValidationError(ApplicationError):
    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")

class NotFoundError(ApplicationError):
    def __init__(self, message: str):
        super().__init__(message, code="NOT_FOUND_ERROR")
```

---

## 📖 Topic 2: Exception Chaining Pattern

```python
import sqlite3

def find_user_by_id(user_id: int):
    try:
        # Lower level database operation
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            raise NotFoundError(f"User with ID #{user_id} was not found.")
        return row[0]
    except sqlite3.Error as err:
        # Chaining low-level database error into high-level DatabaseError
        raise DatabaseError(f"Database query failed for user ID #{user_id}") from err
```

---

## ⚡ Master Cheat Sheet

```python
# Advanced Exception Handling Quick Reference

# 1. Base Exception Definition
class BaseAppError(Exception):
    pass

# 2. Exception Chaining Syntax
try:
    data = parse_json(raw_text)
except ValueError as cause:
    raise FormatError("Invalid JSON input format") from cause

# 3. Accessing Chained Cause
try:
    process_order()
except BaseAppError as e:
    print(f"User message: {e}")
    print(f"Original cause: {e.__cause__}")
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Bare `raise Exception("msg")` without Custom Classes:**
   - ❌ Raising generic `Exception` makes it impossible for callers to selectively handle validation errors vs. database errors.
   - ✅ Always inherit from custom exception classes.

2. **Swallowing the Original Traceback (`raise NewError(...)` without `from`):**
   - ❌ Re-raising without `from cause` loses the root cause traceback during debugging.
   - ✅ Always use `raise CustomError("msg") from original_exception`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between `raise CustomError` and `raise CustomError from original_err`?
**Answer:** Using `from original_err` sets the `__cause__` attribute on the newly raised `CustomError`, chaining the traceback so that developers can trace the exact original root cause of the error.

### Q2: Why inherit from `Exception` instead of `BaseException`?
**Answer:** `BaseException` is the root of all exceptions, including system exits like `SystemExit` and `KeyboardInterrupt` (`Ctrl+C`). Custom application exceptions must inherit from `Exception` so catching them won't prevent the user from interrupting the script.

---

## 📝 Recap Checklist
- [x] Implemented `ApplicationError` root base class with error codes.
- [x] Built domain exceptions: `ValidationError`, `DatabaseError`, `NotFoundError`, `AuthenticationError`, `ExternalServiceError`.
- [x] Used explicit exception chaining (`raise ... from err`).
- [x] Implemented application service layer utilizing the custom error system.
- [x] Created Pytest test suite testing all custom exceptions with 15+ test cases.

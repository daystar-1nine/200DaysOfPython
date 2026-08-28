# 🐍 Day 36/200 – Masterclass Notes: Context Managers & Resource Management

🎯 **Goal:** Understand Python **Context Managers** and the `with` statement protocol—mastering class-based resource management with `__enter__()` and `__exit__()`, exception handling and suppression in `__exit__()`, generator-based context managers with `@contextmanager` from `contextlib`, and building production resource managers for SQLite Database Transactions (`COMMIT` / `ROLLBACK`), Performance Timers, and Temporary File Cleanup.

---

## 📌 Executive Summary & Key Takeaways

- **The `with` Statement Protocol:** Guarantees resource acquisition and cleanup regardless of whether code exits normally or raises an unhandled exception.
- **Class-Based Context Managers:**
  - **`__enter__(self)`:** Acquires the resource, prepares state, and returns the object to be bound to the `as target` variable.
  - **`__exit__(self, exc_type, exc_val, exc_tb)`:** Called automatically upon exiting the `with` block. Returning `True` suppresses any raised exception; returning `False` (or `None`) allows the exception to propagate.
- **`@contextmanager` (Generator Approach):** Simplifies context manager creation using `try...finally` blocks. Everything before `yield` acts as `__enter__()`; everything inside `finally:` acts as `__exit__()`.
- **Database Transaction Management:** Enforces atomicity (ACID principle):
  - On normal block exit $\rightarrow$ `conn.commit()`
  - On exception within block $\rightarrow$ `conn.rollback()`

---

## 📖 Topic 1: Class-Based vs. Generator Context Managers

```python
# 1. Class-Based Approach
class ManagedResource:
    def __enter__(self):
        print("Acquiring resource...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Cleaning up resource...")
        return False  # Propagate exceptions

# 2. Generator-Based Approach using contextlib
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("Acquiring resource...")
    try:
        yield
    finally:
        print("Cleaning up resource...")
```

---

## 📖 Topic 2: Database Transaction Context Manager

```python
import sqlite3

class TransactionManager:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Transaction failed ({exc_val}): ROLLBACK")
            self.connection.rollback()
        else:
            print("Transaction succeeded: COMMIT")
            self.connection.commit()
        return False
```

---

## ⚡ Master Cheat Sheet

```python
# Context Managers Quick Reference

# 1. Basic Class Context Manager Template
class ResourceContext:
    def __enter__(self):
        # Setup / acquire resource
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup / release resource
        return False

# 2. Basic Generator Context Manager Template
from contextlib import contextmanager

@contextmanager
def resource_context():
    # Setup / acquire
    try:
        yield
    finally:
        # Cleanup / release
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Accidentally Suppressing Exceptions in `__exit__()`:**
   - ❌ Returning `True` unconditionally inside `__exit__()` swallows syntax errors, NameErrors, and KeyErrors silently.
   - ✅ Only return `True` when you explicitly intend to suppress a specific handled error.

2. **Forgetting `finally:` Block in `@contextmanager`:**
   - ❌ Code after `yield` without `try...finally` will not execute if an exception occurs during `yield`.
   - ✅ Always place resource release logic inside a `finally:` block.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: How does Python determine whether to suppress an exception raised inside a `with` block?
**Answer:** Python inspects the return value of `__exit__()`. If `__exit__()` returns a truthy value (`True`), the exception is suppressed. If it returns a falsy value (`False` or `None`), Python re-raises the exception.

### Q2: What are the four arguments passed to `__exit__()`?
**Answer:** `self`, `exc_type` (exception class), `exc_val` (exception instance), and `exc_tb` (traceback object). If no exception occurred, the last three arguments are `None`.

---

## 📝 Recap Checklist
- [x] Mastered `__enter__()` and `__exit__()` protocol lifecycle.
- [x] Created class-based context managers for Database Transactions, Performance Timers, and Temporary Files.
- [x] Created generator-based context managers using `@contextmanager`.
- [x] Implemented transaction atomicity (`COMMIT` on success, `ROLLBACK` on error).
- [x] Created Pytest test suite covering normal execution, exceptions, cleanup, commit, rollback, and nested contexts.

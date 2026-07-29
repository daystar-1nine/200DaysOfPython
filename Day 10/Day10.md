# 🐍 Day 10/200 – Masterclass Notes: Exception Handling in Python

🎯 **Goal:** Learn how to handle runtime errors gracefully using `try`, `except`, `else`, `finally`, `raise`, and custom exceptions so your Python applications remain reliable, robust, and crash-resistant.

---

## 📌 Executive Summary & Key Takeaways

- **Runtime Errors vs Syntax Errors:** Syntax errors occur at compile-time before code executes; Exceptions (runtime errors) occur during program execution (e.g. dividing by zero, missing files, invalid type conversions).
- **Graceful Degradation:** Exception handling catches runtime errors before they crash the Python interpreter, allowing fallback logic or user-friendly error messages.
- **The Control Flow:**
  - `try`: Encloses code that *might* throw an exception.
  - `except`: Catches and handles specific thrown exceptions.
  - `else`: Executes *only if no exceptions occurred* in the `try` block.
  - `finally`: ALWAYS executes regardless of whether exceptions were thrown or caught (ideal for cleanup operations like closing files or database connections).
- **Custom Exceptions:** Extend Python's base `Exception` class to create domain-specific error types (e.g. `InvalidAgeError`, `InsufficientFundsError`).

---

## 📖 Topic 1: Understanding Exceptions

### 1.1 Common Built-in Python Exceptions

| Exception Name | Trigger Cause | Real-World Example |
|---|---|---|
| `ValueError` | Correct data type but invalid value | `int("abc")` |
| `ZeroDivisionError` | Division or modulo by zero | `10 / 0` |
| `IndexError` | Sequence subscript index out of range | `lst = [1, 2]; lst[5]` |
| `KeyError` | Mapping key not found in dictionary | `d = {"a": 1}; d["b"]` |
| `FileNotFoundError` | Target file path does not exist on disk | `open("missing.txt")` |
| `TypeError` | Operation applied to incompatible types | `"age: " + 25` |
| `NameError` | Variable referenced before declaration | `print(undefined_var)` |
| `AttributeError` | Object reference lacks invoked attribute/method | `"hello".append("x")` |

---

## 📖 Topic 2: The `try`, `except`, `else`, `finally` Lifecycle

### 2.1 Complete Lifecycle Architecture

```python
try:
    # Code that might raise an exception
    num = int(input("Enter divisor: "))
    result = 100 / num
except ValueError:
    # Executes ONLY if a ValueError is raised
    print("Error: Input must be an integer!")
except ZeroDivisionError:
    # Executes ONLY if a ZeroDivisionError is raised
    print("Error: Division by zero is mathematically undefined!")
else:
    # Executes ONLY if NO exception was raised in try block
    print(f"Success! Result = {result}")
finally:
    # ALWAYS executes regardless of success or failure
    print("Cleanup: Execution block complete.")
```

---

## 📖 Topic 3: Raising & Creating Custom Exceptions

### 3.1 Raising Standard Exceptions with `raise`

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    if age > 120:
        raise ValueError("Age exceeds realistic human limit!")
    return f"Age {age} accepted."
```

### 3.2 Creating Custom Exception Classes

```python
class InsufficientBalanceError(Exception):
    """Custom exception raised when withdrawal exceeds available balance."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Attempted to withdraw Rs.{amount}, but balance is only Rs.{balance}.")

# Usage
balance = 1000
withdraw = 1500

if withdraw > balance:
    raise InsufficientBalanceError(balance, withdraw)
```

---

## ⚡ Master Cheat Sheet & Quick Summary

```python
# Exception Handling Cheat Sheet

# 1. Catching Multiple Exception Types Together
try:
    val = int("abc")
except (ValueError, TypeError) as e:
    print(f"Caught expected input error: {e}")

# 2. Accessing Exception Details
try:
    res = 10 / 0
except ZeroDivisionError as err:
    print(f"Error Type: {type(err).__name__} | Message: {err}")

# 3. Clean Exception-Reraising
try:
    process_data()
except Exception as e:
    logger.error("Processing failed", exc_info=True)
    raise  # Re-raises the caught exception
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Bare `except:` Clauses:**
   - ❌ `except:` (Catches everything including `KeyboardInterrupt` and `SystemExit`, making programs hard to stop).
   - ✅ Always specify the exception type: `except (ValueError, KeyError):`.

2. **Swallowing Exceptions Silently:**
   - ❌ `except Exception: pass` (Hides bugs and leaves the application in an indeterminate state).
   - ✅ Log or handle the exception explicitly.

3. **Using Exceptions for Normal Control Flow:**
   - ❌ Using `try-except` to check if a key exists in a dictionary when `dict.get(key)` or `if key in dict:` is cleaner.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What happens if an exception is raised inside the `else` block?
**Answer:** An exception raised in the `else` block is NOT caught by the `except` blocks of the same `try-except` statement. It propagates up the call stack unless caught by an outer `try-except` wrapper.

### Q2: Will the `finally` block execute if the `try` or `except` block contains a `return` statement?
**Answer:** YES! The `finally` block is guaranteed to execute *before* the function actually returns control to the caller, even if a `return`, `break`, or `continue` statement is executed inside `try` or `except`.

---

## 📝 Recap Checklist
- [x] Distinguished between syntax errors and runtime exceptions.
- [x] Implemented single and multiple `except` handlers.
- [x] Used `else` for clean happy-path logic and `finally` for mandatory resource cleanup.
- [x] Raised custom exception classes inheriting from Python's base `Exception`.

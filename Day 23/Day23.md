# 🐍 Day 23/200 – Masterclass Notes: Logging & Debugging in Python

🎯 **Goal:** Master professional application logging using Python's `logging` module, configure custom loggers/handlers/formatters, log exception tracebacks (`exc_info=True`), interpret error tracebacks, and use `breakpoint()` / `pdb` for interactive debugging.

---

## 📌 Executive Summary & Key Takeaways

- **Why Logging vs. `print()`:** `print()` statements are ephemeral and lost when terminal sessions close. Proper logging captures timestamps, severity levels, process names, and stack tracebacks directly into persistent log files.
- **Log Levels Hierarchy:**
  - `DEBUG` (10): Granular diagnostics for developers.
  - `INFO` (20): Normal application state events (startup, login, completion).
  - `WARNING` (30): Unexpected events or potential issues (low storage, retry attempt).
  - `ERROR` (40): Application feature failure (payment declined, file missing).
  - `CRITICAL` (50): System-wide failure (database connection lost, fatal crash).
- **Logger Architecture:**
  - `Logger`: Main API object (`logging.getLogger(__name__)`).
  - `Handler`: Directs log streams (`FileHandler`, `StreamHandler`).
  - `Formatter`: Specifies output log structure (`%(asctime)s | %(levelname)s | %(name)s | %(message)s`).
- **Interactive Debugging (`breakpoint()`):** Invokes built-in Python debugger `pdb` to step through code execution dynamically.

---

## 📖 Topic 1: Standard `logging` Module Fundamentals

### 1.1 Basic Configuration & File Logging

```python
import logging

# Basic File Configuration
logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.info("Application started successfully")
logging.warning("Memory utilization is above 80%")
logging.error("Failed to connect to secondary caching server")
```

### 1.2 Exception Stack Traceback Logging

```python
import logging

try:
    result = 10 / 0
except ZeroDivisionError:
    # logging.exception automatically appends full traceback
    logging.exception("Division operation failed due to zero denominator")
```

---

## 📖 Topic 2: Professional Logger & Handler Architecture

```python
import logging

def setup_logger(logger_name: str, log_file: str = "app.log") -> logging.Logger:
    """Factory function creating a modular logger with FileHandler and StreamHandler."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if logger is instantiated multiple times
    if not logger.handlers:
        # File Handler (logs INFO and above to file)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # Console Handler (logs DEBUG and above to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
```

---

## 📖 Topic 3: Interactive Debugging (`pdb` / `breakpoint()`)

### 3.1 Debugger Commands Reference

| Command | Action | Description |
|---|---|---|
| `p variable` | Print | Prints the current value of `variable` |
| `n` | Next | Executes current line and advances to next line in same function |
| `s` | Step | Steps into the function call on the current line |
| `c` | Continue | Continues execution until next breakpoint or program completion |
| `l` | List | Displays source code surrounding current line |
| `q` | Quit | Instantly aborts debugger and program execution |

---

## ⚡ Master Cheat Sheet

```python
# Logging & Debugging Master Cheat Sheet

import logging

# 1. Custom Logger Setup
logger = logging.getLogger(__name__)

# 2. Variable Formatting in Logging (Avoid manual string concatenation)
user_id = 42
logger.info("User %s successfully authenticated", user_id)

# 3. Exception Logging with Stack Trace
try:
    val = int("invalid_num")
except ValueError as e:
    logger.error("Failed to parse integer: %s", e, exc_info=True)

# 4. Built-in Breakpoint
# breakpoint()  # Opens interactive pdb terminal
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Using Manual String Formatting in Log Calls:**
   - ❌ `logging.info(f"User {name} logged in")` (Evaluates string even if log level is disabled).
   - ✅ `logging.info("User %s logged in", name)` (Lazy evaluation; formatted only if log level is active).

2. **Swallowing Exceptions Without Logging Stack Trace:**
   - ❌ `except Exception: print("Error occurred")` (Hides root cause traceback).
   - ✅ `except Exception: logger.exception("Unhandled error occurred")`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between `logging.error()` and `logging.exception()`?
**Answer:** `logging.error()` logs a message at `ERROR` level without a stack trace unless `exc_info=True` is explicitly passed. `logging.exception()` logs at `ERROR` level and automatically attaches the full exception traceback. It must be called within an `except` block.

### Q2: What is the benefit of `logging.getLogger(__name__)` over `logging.info()`?
**Answer:** `logging.getLogger(__name__)` creates a module-scoped logger named after the Python module hierarchy (e.g. `utils.database`). This allows fine-grained log filtering and handler routing per module rather than dumping all logs into the global root logger.

---

## 📝 Recap Checklist
- [x] Configured basic and file-based logging (`basicConfig`).
- [x] Applied all 5 standard log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- [x] Created custom loggers, `FileHandler`, `StreamHandler`, and custom log formatters.
- [x] Captured exception tracebacks using `logging.exception()` and `exc_info=True`.
- [x] Used `breakpoint()` for interactive debugging and traceback inspection.
- [x] Built Application Logger, ATM System with Logging, and Expense Tracker with Logging projects.

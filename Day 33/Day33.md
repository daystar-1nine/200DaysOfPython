# 🐍 Day 33/200 – Masterclass Notes: Python Decorators & Function Monitoring

🎯 **Goal:** Master Python **Decorators**—understanding First-Class Functions, Nested Functions, Closures, `@` syntactic sugar, wrapper parameters (`*args`, `**kwargs`), preserving function metadata (`functools.wraps`), building Parameterized Decorators (`@retry(max_attempts=3)`), and stacking multiple decorators.

---

## 📌 Executive Summary & Key Takeaways

- **First-Class Functions:** In Python, functions are first-class objects. They can be assigned to variables, passed as arguments into other functions, and returned from functions.
- **Closures & Nested Functions:** A decorator is a higher-order function that takes a target function, defines an inner wrapper function that captures the target function (closure), and returns the wrapper.
- **The `@` Syntactic Sugar:**
  ```python
  @logger
  def hello():
      pass
  # Equivalent to: hello = logger(hello)
  ```
- **Metadata Preservation (`@wraps(func)`):** Always decorate inner wrapper functions with `@wraps(func)` from `functools` to preserve `__name__`, `__doc__`, and function annotations.
- **Decorator Stacking Order:** When applying multiple decorators (`@timer` above `@logger`), decorators execute from the bottom up during application (outer wrapper = `@timer`, inner wrapper = `@logger`).

---

## 📖 Topic 1: Standard Wrapper Pattern

```python
from functools import wraps

def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function: '{func.__name__}'")
        result = func(*args, **kwargs)
        print(f"Finished function: '{func.__name__}'")
        return result
    return wrapper
```

---

## 📖 Topic 2: Parameterized Decorators (Decorator Factories)

```python
from functools import wraps
import time

def retry(max_attempts: int = 3, delay: float = 0.1):
    """Decorator factory returning a parameterized decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
```

---

## ⚡ Master Cheat Sheet

```python
# Decorators Master Cheat Sheet

# 1. Basic Decorator Template
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Pre-execution logic
        res = func(*args, **kwargs)
        # Post-execution logic
        return res
    return wrapper

# 2. Parameterized Decorator Template
def my_factory(param1, param2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use param1, param2
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `@wraps(func)`:**
   - ❌ Omitting `@wraps(func)` causes decorated function's `__name__` to become `'wrapper'`, breaking docstrings and debugging tools.
   - ✅ Always add `@wraps(func)` to wrapper functions.

2. **Forgetting to Return Function Result in Wrapper:**
   - ❌ `func(*args, **kwargs)` executed without `return result` makes decorated function return `None`.
   - ✅ Always capture and return `result = func(*args, **kwargs)`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the execution order when multiple decorators are stacked on a function?
**Answer:** Decorators wrap the function from bottom to top. For `@timer` over `@logger`, calling the decorated function enters `@timer` wrapper first, which calls `@logger` wrapper, which finally calls the target function.

### Q2: How do you create a decorator that accepts arguments?
**Answer:** By wrapping the decorator inside an outer "factory" function that accepts parameters and returns the actual decorator function.

---

## 📝 Recap Checklist
- [x] Mastered First-Class Functions and Closures.
- [x] Created standard `@logger` and `@timer` decorators with `@wraps`.
- [x] Created parameterized `@retry(max_attempts=3)` decorator.
- [x] Created authorization decorator `@requires_auth(role="admin")`.
- [x] Built the Function Monitoring System codebase with 12+ Pytest test cases.

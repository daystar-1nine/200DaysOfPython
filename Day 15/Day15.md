# 🐍 Day 15/200 – Masterclass Notes: Decorators & Closures

🎯 **Goal:** Master higher-order functions, **Closures**, the `@decorator` syntax, `*args` / `**kwargs` wrappers, preserving function metadata using `functools.wraps`, stacked decorators, and enterprise use-cases (Logging, Authentication, and Execution Timing).

---

## 📌 Executive Summary & Key Takeaways

- **First-Class Objects:** In Python, functions are first-class citizens—they can be stored in variables, passed as arguments to other functions, and returned from functions.
- **Nested Functions & Scoping:** Inner functions can access variables from their outer enclosing scope (LEGB Scoping: Local $\rightarrow$ Enclosing $\rightarrow$ Global $\rightarrow$ Built-in).
- **Closures:** A closure is an inner function that retains access to variables from its enclosing scope even after the outer function has finished executing.
- **Decorators:** A decorator is a design pattern that wraps a target function to extend or modify its behavior dynamically without altering the original function's source code.
- **`@functools.wraps`:** Essential decorator utility that preserves the original wrapped function's `__name__`, `__doc__`, and signature metadata.

---

## 📖 Topic 1: Functions as First-Class Objects & Closures

### 1.1 First-Class Citizens in Python

```python
def uppercase_transformer(text):
    return text.upper()

# Assigning function to a variable
func_ref = uppercase_transformer
print(func_ref("hello"))  # "HELLO"

# Passing function as an argument
def apply_transform(func, data):
    return func(data)

print(apply_transform(uppercase_transformer, "python"))  # "PYTHON"
```

### 1.2 Closures Deep Dive (`nonlocal` & Cell Objects)

```python
def make_counter(start_value=0):
    count = start_value  # Enclosing variable

    def increment():
        nonlocal count  # Declares intent to mutate enclosing variable
        count += 1
        return count

    return increment

counter1 = make_counter(10)
print(counter1())  # 11
print(counter1())  # 12
```
*How it works:* The variable `count` is stored in `counter1.__closure__[0].cell_contents`, keeping state alive in memory even though `make_counter()` has returned!

---

## 📖 Topic 2: Decorator Mechanics & The `@` Syntactic Sugar

### 2.1 Anatomy of a Python Decorator

```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # Preserves docstrings and __name__
    def wrapper(*args, **kwargs):
        print("1. [Pre-Execution] Setup logic here")
        result = func(*args, **kwargs)  # Call original function
        print("2. [Post-Execution] Cleanup logic here")
        return result
    return wrapper

# Applying via syntactic sugar @
@my_decorator
def calculate_sum(a, b):
    """Calculates sum of two numbers."""
    return a + b

# Equivalent to: calculate_sum = my_decorator(calculate_sum)
print(calculate_sum(10, 20))  # 30
print(calculate_sum.__name__)  # 'calculate_sum' (Thanks to @wraps)
```

---

## 📖 Topic 3: Stacking Multiple Decorators

When applying multiple decorators:
```python
@decorator_one
@decorator_two
def my_function():
    pass
```
This is evaluated bottom-up as:
`my_function = decorator_one(decorator_two(my_function))`

1. `decorator_two` wraps `my_function`.
2. `decorator_one` wraps the wrapper returned by `decorator_two`.

---

## ⚡ Master Cheat Sheet

```python
# Decorators & Closures Cheat Sheet

import functools
import time

# 1. Universal Function Timer Decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] Executed in {elapsed:.6f}s")
        return res
    return wrapper

# 2. Parameterized Decorator Factory
def repeat(num_times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                res = func(*args, **kwargs)
            return res
        return wrapper
    return decorator
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `*args` and `**kwargs` in Wrapper:**
   - ❌ `def wrapper(): return func()` (Fails when decorating functions taking arguments with `TypeError`).
   - ✅ Always define `def wrapper(*args, **kwargs): return func(*args, **kwargs)`.

2. **Forgetting `@functools.wraps(func)`:**
   - Omitting `@wraps` causes decorated functions to lose their name (`__name__` becomes `'wrapper'`) and docstrings, breaking introspection tools and unit test runners.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between a Function and a Closure in Python?
**Answer:** A regular function only has access to its local parameters/variables and global scope. A **Closure** is a nested function that retains bound references to non-local variables from its outer enclosing scope even after the outer function's stack frame has been popped.

### Q2: How do you pass arguments to a Decorator itself (Parameterized Decorator)?
**Answer:** You create a **Decorator Factory**—a function that accepts arguments and returns the actual decorator function, which in turn returns the wrapper function (3 levels of nested functions).

---

## 📝 Recap Checklist
- [x] Treated functions as first-class objects (passing, returning, assigning).
- [x] Built Closures preserving state via `nonlocal` and enclosing scopes.
- [x] Written custom decorators using `@decorator` syntactic sugar.
- [x] Supported arbitrary arguments using `*args` and `**kwargs`.
- [x] Preserved function signatures using `@functools.wraps`.
- [x] Built production decorators for Timing, Logging, and Authentication.

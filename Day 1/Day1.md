# 🐍 Day 1/200 – Masterclass Notes: Python Basics, Variables, Setup & Naming Conventions

🎯 **Goal:** Master the fundamentals of Python programming, environment setup, input/output operations, dynamic typing, and industry-standard naming conventions.

---

## 📌 Executive Summary & Key Takeaways

- **Python Character:** High-level, interpreted, dynamically typed, garbage-collected, multi-paradigm programming language.
- **Dynamic Typing:** Variables do not require explicit type declaration; types are bound to values at runtime.
- **PEP 8:** The official Python style guide enforcing 4-space indentation, `snake_case` variable/function names, and clear readability.
- **Input & Output:** `print()` for formatted output and `input()` (which **always** returns a string).

---

## 📖 Topic 1: What is Python & How it Works

### 1.1 Key Characteristics of Python
1. **Interpreted:** Python source code (`.py`) is compiled into bytecode (`.pyc`) by CPython, which is then executed by the Python Virtual Machine (PVM).
2. **Dynamically Typed:** Variable types are determined at runtime, not compile time.
3. **High-Level & Garbage Collected:** Automatic memory management via reference counting and a generational garbage collector.

```text
[Source Code: script.py] ──> (CPython Compiler) ──> [Bytecode: script.pyc] ──> (PVM / Interpreter) ──> [Execution Output]
```

---

## 📖 Topic 2: Variables & Dynamic Typing

### 2.1 What is a Variable?
In Python, a variable is **not a container storing data**, but a **labeled pointer/reference** pointing to an object stored in memory.

```python
x = 10
# Memory View: Variable 'x' is a pointer referencing the integer object 10 in memory.

# Identity Check: id() returns the unique memory address of an object
print(id(x))
```

### 2.2 Rebinding & Dynamic Typing
Because variables are references, you can rebind a variable name to an object of a completely different data type at any time.

```python
age = 20          # Bound to int object
print(type(age))  # <class 'int'>

age = "Twenty"    # Rebound to str object
print(type(age))  # <class 'str'>
```

---

## 📖 Topic 3: Naming Conventions (PEP 8 Standards)

| Entity | Naming Style | Example |
|---|---|---|
| **Variables & Functions** | `snake_case` | `user_age`, `calculate_total()` |
| **Constants** | `ALL_CAPS` | `MAX_CONNECTIONS = 100`, `PI = 3.14` |
| **Classes** | `PascalCase` | `StudentManager`, `DatabaseConnection` |
| **Modules / Files** | `snake_case` | `print_hello.py`, `data_type.py` |

### ❌ Illegal Variable Names
- `1st_name = "Suraj"` (Cannot start with a number)
- `user-name = "Suraj"` (Hyphens `-` are invalid; treated as subtraction)
- `class = "Python"` (Cannot use Python reserved keywords like `class`, `def`, `if`, `for`)

---

## 📖 Topic 4: Input and Output Operations

### 4.1 Printing Output (`print()`)
The `print()` function formats and writes output to standard stdout.

```python
name = "Suraj"
age = 20

# 1. String Concatenation (+)
print("Hello " + name)

# 2. Multiple arguments (comma-separated, auto-spaced)
print("Name:", name, "| Age:", age)

# 3. f-Strings (Formatted String Literals - Recommended!)
print(f"Hello {name}, you are {age} years old.")

# 4. Custom Separator (sep) and End Character (end)
print("Python", "Java", "C++", sep=" | ")      # Output: Python | Java | C++
print("Loading", end="...")                   # Does not print newline
print("Done!")                                 # Output: Loading...Done!
```

### 4.2 User Input (`input()`)
The `input()` function pauses execution, reads a line from stdin, and **always returns it as a string (`str`)**.

```python
# ⚠️ CRITICAL: input() returns a string!
age_str = input("Enter your age: ")
print(type(age_str))  # <class 'str'>

# ✅ Type casting input to integer or float
age = int(input("Enter your age: "))
print(f"Next year you will be {age + 1} years old.")
```

---

## ⚡ Master Cheat Sheet & Quick Summary

```python
# Variables & Output
name = "Suraj"                   # String variable (snake_case)
age = 20                         # Integer variable
height = 5.9                     # Float variable
is_student = True                # Boolean variable

# Formatted Output
print(f"Student {name} (Age: {age}, Height: {height}ft)")

# Type Conversion
user_input = "42"
num = int(user_input)            # Casts "42" to integer 42
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting Type Casting on `input()`:**
   - ❌ `sum = input() + input()` (Performs string concatenation `"5" + "10" = "510"`)
   - ✅ `sum = int(input()) + int(input())` (Performs numeric addition `5 + 10 = 15`)

2. **Using Reserved Keywords:**
   - ❌ `str = "hello"` (Overwrites Python's built-in `str()` type constructor!)
   - ✅ `text_str = "hello"`

3. **Indentation Errors:**
   - Python uses strict indentation (4 spaces per block level) instead of curly braces `{}`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What happens under the hood when you execute `a = 5` followed by `b = a`?
**Answer:** Python creates an integer object `5` in memory. `a` is set to point to `5`. When `b = a` is executed, `b` is assigned to point to the exact same integer object `5` in memory (referencing the same `id(5)`).

### Q2: Why is Python called a dynamically typed language?
**Answer:** Because variable types are evaluated dynamically at runtime rather than explicitly declared before compilation.

### Q3: How do you print multiple statements on the same line in Python?
**Answer:** Pass the `end=""` keyword argument to `print()`:
```python
print("Hello ", end="")
print("World!")  # Output: Hello World!
```

---

## 📝 Recap Checklist
- [x] Installed Python environment & understand `.py` script execution.
- [x] Mastered `snake_case` naming conventions and PEP 8 guidelines.
- [x] Know how `print()` with f-strings and custom `sep`/`end` works.
- [x] Understand that `input()` always returns a `str` and requires explicit casting for math.

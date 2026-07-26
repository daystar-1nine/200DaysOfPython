# 🐍 Day 5/200 – Masterclass Notes: Functions, Arguments, Scope & Return Values

🎯 **Goal:** Master modular programming in Python using functions, positional/keyword arguments, default parameters, variable-length parameters (`*args`, `**kwargs`), `return` statements, and variable scope (LEGB rule).

---

## 📌 Executive Summary & Key Takeaways

- **Function:** A reusable block of organized code designed to perform a specific task (enforces DRY principle).
- **Parameters vs Arguments:** Parameters are variable placeholders in the function definition; Arguments are actual values passed during invocation.
- **`return` Statement:** Sends a computed value back to the caller and terminates function execution (defaults to `None`).
- **LEGB Scope Rule:** Python resolves variable names in this order: **L**ocal $\rightarrow$ **E**nclosing $\rightarrow$ **G**lobal $\rightarrow$ **B**uilt-in.

---

## 📖 Topic 1: Function Fundamentals & Syntax

### 1.1 Defining and Calling Functions
Functions are defined using the `def` keyword, followed by function name, parameter parentheses `()`, docstring, and indented body block.

```python
def greet():
    """Docstring: Prints a standard welcome greeting."""
    print("Hello, Welcome to Python Functions!")

# Invoking / Calling the function
greet()
```

---

## 📖 Topic 2: Parameters, Arguments & Default Values

### 2.1 Positional & Keyword Arguments

```python
def describe_person(name, age):
    print(f"Name: {name}, Age: {age}")

# Positional Arguments (Order matters!)
describe_person("Suraj", 20)

# Keyword Arguments (Order does NOT matter!)
describe_person(age=20, name="Suraj")
```

### 2.2 Default Parameters ⚠️
Default parameters provide fallback values if an argument is omitted during function call.

```python
# ⚠️ Default parameters MUST follow positional parameters in the function header!
def greet_user(name, role="Guest"):
    print(f"Hello {name}, your role is {role}.")

greet_user("Suraj")          # Uses default: Hello Suraj, your role is Guest.
greet_user("Suraj", "Admin") # Overrides default: Hello Suraj, your role is Admin.
```

---

## 📖 Topic 3: Flexible Arguments (`*args` and `**kwargs`)

When you don't know in advance how many arguments will be passed:

1. **`*args` (Arbitrary Positional Arguments):** Packs extra positional arguments into a **tuple**.
2. **`**kwargs` (Arbitrary Keyword Arguments):** Packs extra keyword arguments into a **dictionary**.

```python
def calculate_sum(*args):
    """Calculates sum of any number of positional arguments."""
    print("args tuple:", args)
    return sum(args)

print(calculate_sum(10, 20, 30, 40))  # Output: 100

def print_user_profile(**kwargs):
    """Prints arbitrary key-value metadata."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_user_profile(name="Suraj", age=20, country="India")
```

---

## 📖 Topic 4: The `return` Statement

A function without an explicit `return` statement implicitly returns `None`.

```python
def square(num):
    return num * num  # Returns computed value back to caller

result = square(5)
print(result)  # 25

# Returning multiple values (Returned as a tuple!)
def get_min_max(numbers):
    return min(numbers), max(numbers)

minimum, maximum = get_min_max([10, 5, 20, 85, 3])
print(f"Min: {minimum}, Max: {maximum}")  # Min: 3, Max: 85
```

---

## 📖 Topic 5: Variable Scope & LEGB Rule

Scope determines where in the program a variable can be accessed.

```text
[ Local (Inside Function) ] ──> [ Enclosing (Outer Function) ] ──> [ Global (Module Level) ] ──> [ Built-in (Python keywords) ]
```

```python
x = "Global Variable"

def outer_function():
    x = "Enclosing Variable"
    
    def inner_function():
        x = "Local Variable"
        print("Inner:", x)  # Prints: Local Variable
        
    inner_function()
    print("Outer:", x)      # Prints: Enclosing Variable

outer_function()
print("Global:", x)          # Prints: Global Variable
```

### Modifying Global Variables (`global` keyword)
```python
counter = 0

def increment():
    global counter  # Declares intent to modify global counter variable
    counter += 1

increment()
print(counter)  # Output: 1
```

---

## ⚡ Master Cheat Sheet & Quick Summary

```python
# Function Signature Cheat Sheet
def complete_function_demo(pos1, pos2, default_param="Default", *args, **kwargs):
    """
    Order of Parameters:
    1. Positional parameters
    2. Default parameters
    3. *args
    4. **kwargs
    """
    pass
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Mutable Default Arguments Trap ❌:**
   - ❌ Never use mutable objects (`list`, `dict`) as default argument values!
     ```python
     def append_item(item, target_list=[]): # Dangerous! Shares same list across calls
         target_list.append(item)
         return target_list
     ```
   - ✅ Use `None` as the default value instead:
     ```python
     def append_item(item, target_list=None):
         if target_list is None:
             target_list = []
         target_list.append(item)
         return target_list
     ```

2. **Docstrings:** Always write descriptive triple-quoted `"""docstrings"""` for public functions.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What does `*args` and `**kwargs` do in a Python function signature?
**Answer:** `*args` captures excess positional arguments into a `tuple`. `**kwargs` captures excess keyword arguments into a `dict`.

### Q2: What happens if a function does not have a `return` statement?
**Answer:** The function automatically returns `None` upon reaching the end of its execution block.

### Q3: What is the LEGB rule in Python variable scoping?
**Answer:** LEGB stands for **L**ocal, **E**nclosing, **G**lobal, and **B**uilt-in. It defines the hierarchy Python follows when searching for variable names.

---

## 📝 Recap Checklist
- [x] Defined functions using `def` with parameters, return statements, and docstrings.
- [x] Mastered positional vs keyword arguments and default parameter positioning.
- [x] Leveraged `*args` and `**kwargs` for dynamic inputs.
- [x] Understood the LEGB scope hierarchy and avoided mutable default argument bugs.

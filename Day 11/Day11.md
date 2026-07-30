# 🐍 Day 11/200 – Masterclass Notes: Modules, Packages & the Python Standard Library

🎯 **Goal:** Learn how to structure code modularly into reusable modules (`.py` files) and packages (folder hierarchies), leverage Python's rich built-in Standard Library (`math`, `random`, `datetime`, `os`, `sys`, `statistics`, `string`), and manage external third-party packages using `pip`.

---

## 📌 Executive Summary & Key Takeaways

- **Modular Design:** A module is a single `.py` file containing functions, variables, and classes. Modular programming promotes code reusability, maintainability, and namespace isolation.
- **Packages:** A directory containing multiple `.py` modules and an optional `__init__.py` marker file.
- **The Standard Library:** Python comes "batteries included" with powerful built-in modules so you don't need to reinvent fundamental utilities like date math, random number generation, system paths, or statistical summaries.
- **Package Management (`pip`):** The standard package installer for Python that downloads third-party libraries from PyPI (Python Package Index).

---

## 📖 Topic 1: What is a Module & Import Syntax

### 1.1 Four Ways to Import Modules

```python
# 1. Standard Module Import (Recommended for clarity)
import math
print(math.sqrt(25))  # Requires module prefix 'math.'

# 2. Importing Specific Names directly into current namespace
from math import sqrt, pi
print(sqrt(25))       # No prefix needed
print(pi)

# 3. Importing with an Alias (Convenient for long module names)
import datetime as dt
now = dt.datetime.now()

# 4. Wildcard Import (⚠️ DISCOURAGED in production code)
from math import *     # Pollution of global namespace!
```

---

## 📖 Topic 2: Creating Custom Modules & Packages

### 2.1 Package Directory Structure

```text
my_project/
├── main.py
└── utilities/               # Package directory
    ├── __init__.py          # Marks directory as Python package
    ├── math_helper.py       # Sub-module
    └── string_helper.py     # Sub-module
```

### 2.2 Importing from a Package
```python
# In main.py:
from utilities.math_helper import calculate_area
from utilities.string_helper import clean_text
```

---

## 📖 Topic 3: Deep Dive into Built-in Standard Library Modules

### 3.1 `math` Module (Mathematical Calculations)
- `math.sqrt(x)`: Returns square root of $x$.
- `math.factorial(n)`: Computes $n!$.
- `math.ceil(x)` / `math.floor(x)`: Rounding up/down.
- `math.pi`, `math.e`: Mathematical constants.

### 3.2 `random` Module (Pseudo-Random Generation)
- `random.randint(a, b)`: Random integer in $[a, b]$.
- `random.choice(sequence)`: Pick a random item from a list/tuple.
- `random.shuffle(list)`: Shuffle list items in-place.
- `random.sample(population, k)`: Pick $k$ unique random elements.

### 3.3 `datetime` Module (Date and Time Math)
- `datetime.now()`: Current local timestamp object.
- `datetime.strftime(format)`: Format datetime to string (e.g. `"%Y-%m-%d %H:%M:%S"`).
- `datetime.strptime(date_str, format)`: Parse string date into datetime object.
- `timedelta`: Duration calculations (subtracting two dates yields a `timedelta`).

### 3.4 `os` & `sys` Modules (System & Environment Interaction)
- `os.getcwd()`: Get current working directory path.
- `os.listdir(path)`: List directory contents.
- `os.makedirs(path, exist_ok=True)`: Create nested directory tree.
- `sys.version`: Installed Python version info.
- `sys.path`: Search paths used by Python when locating modules.

### 3.5 `statistics` Module (Statistical Summaries)
- `statistics.mean(data)`: Arithmetic mean.
- `statistics.median(data)`: Middle value.
- `statistics.mode(data)`: Most common value.
- `statistics.stdev(data)`: Sample standard deviation.

---

## ⚡ Master Cheat Sheet

```python
# Quick Standard Library Reference

import math, random, statistics, os
from datetime import datetime, date

# Date math (Days until target date)
today = date.today()
target = date(2026, 12, 31)
days_remaining = (target - today).days

# Random choices
winner = random.choice(["Alice", "Bob", "Charlie"])
die_roll = random.randint(1, 6)

# Statistics
data = [10, 20, 30, 40, 50]
avg = statistics.mean(data)
med = statistics.median(data)

# Safe directory creation
os.makedirs("output/logs", exist_ok=True)
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Shadowing Standard Library Module Names:**
   - ❌ Creating a file named `random.py` or `math.py` in your project folder. When you run `import random`, Python imports YOUR local file instead of the built-in module!
   - ✅ Always give unique names to your custom modules (e.g. `my_random_utils.py`).

2. **Using `from module import *`:**
   - Overwrites existing functions/variables in your scope silently and makes code hard to read/debug.

3. **Circular Imports:**
   - Module A imports Module B, and Module B imports Module A. Solve by refactoring shared dependencies into a separate third module.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the purpose of `__init__.py` in a Python package?
**Answer:** `__init__.py` marks a directory as a Python package. In Python 3.3+, implicit namespace packages exist without `__init__.py`, but including `__init__.py` is still best practice for package initialization and specifying exported API symbols via `__all__`.

### Q2: What is `if __name__ == "__main__":` used for in custom modules?
**Answer:** It checks whether the module is being executed directly as the main script or imported into another file. Code inside this block executes ONLY when the script is run directly, preventing test/demo code from executing upon import.

---

## 📝 Recap Checklist
- [x] Mastered `import`, `from...import`, and alias (`as`) syntax.
- [x] Built custom modules and imported them cleanly.
- [x] Utilized `math`, `random`, `datetime`, `os`, `sys`, and `statistics`.
- [x] Managed package installations using `pip`.

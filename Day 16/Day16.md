# 🐍 Day 16/200 – Masterclass Notes: Lambda Functions, map(), filter(), reduce() & Comprehensions

🎯 **Goal:** Master Python's functional programming toolkit including **Lambda Functions**, **`map()`**, **`filter()`**, **`reduce()`**, **List Comprehensions**, **Dictionary Comprehensions**, **Set Comprehensions**, and **Generator Expressions** for writing expressive, pythonic, and clean code.

---

## 📌 Executive Summary & Key Takeaways

- **Lambda Functions:** Anonymous inline functions created with the syntax `lambda arguments: expression`. They are restricted to a single expression and return implicitly.
- **`map(function, iterable)`:** Applies a transformation function to every item in an iterable lazily, returning a map iterator object.
- **`filter(function, iterable)`:** Filters elements from an iterable for which `function(item)` evaluates to `True`, returning a filter iterator object.
- **`reduce(function, sequence)`:** Located in `functools` module; recursively applies a 2-argument function to elements of a sequence to aggregate them into a single accumulated result.
- **Comprehensions:** Declarative syntax construct for building new data structures (`List`, `Dict`, `Set`) cleanly without writing explicit `for` loops and `append()` statements.

---

## 📖 Topic 1: Lambda Functions (Anonymous Functions)

### 1.1 Anatomy & Syntax

```python
# Regular Function
def add_ten(x):
    return x + 10

# Equivalent Lambda Function
add_ten_lambda = lambda x: x + 10

print(add_ten_lambda(5))  # Output: 15

# Lambda with multiple arguments
multiply = lambda a, b: a * b
print(multiply(4, 5))     # Output: 20

# Lambda with conditional ternary expression
parity_check = lambda x: "Even" if x % 2 == 0 else "Odd"
print(parity_check(7))    # Output: "Odd"
```

---

## 📖 Topic 2: Higher-Order Functions (`map`, `filter`, `reduce`)

### 2.1 `map()` - Data Transformation

Transforming every element in a sequence:
```python
numbers = [1, 2, 3, 4, 5]
# Double every number
doubled = list(map(lambda x: x * 2, numbers))
# [2, 4, 6, 8, 10]
```

### 2.2 `filter()` - Conditional Selection

Selecting elements based on boolean predicate:
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
# Keep only even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4, 6, 8]
```

### 2.3 `reduce()` - Aggregation

Accumulating sequence down to a single value:
```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
# Compute total product: (((1 * 2) * 3) * 4) * 5
product = reduce(lambda acc, x: acc * x, numbers)
# Output: 120
```

---

## 📖 Topic 3: Comprehensions

### 3.1 Syntax Comparison Matrix

| Construction Type | Syntax Pattern | Example Output |
|---|---|---|
| **List Comprehension** | `[expr for x in seq if cond]` | `[0, 4, 16, 36]` (List) |
| **Dict Comprehension** | `{k_expr: v_expr for x in seq}` | `{0: 0, 1: 1, 2: 4}` (Dict) |
| **Set Comprehension** | `{expr for x in seq if cond}` | `{1, 2, 3}` (Unique Set) |
| **Generator Expression** | `(expr for x in seq)` | `<generator object>` (Lazy Stream) |

### 3.2 Dictionary & Set Comprehension Examples

```python
# Dictionary Comprehension (Mapping number to its square)
squares_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Set Comprehension (Deduplication + Transformation)
raw_names = ["suraj", "RAHUL", "suraj", "AMIT"]
unique_titles = {name.title() for name in raw_names}
# {'Suraj', 'Rahul', 'Amit'}
```

---

## ⚡ Master Cheat Sheet

```python
# Functional Programming Cheat Sheet

from functools import reduce

# 1. Map, Filter, Reduce pipeline
data = [10, 15, 20, 25, 30]

# Pipeline: Add 5 to even numbers and sum total
evens = filter(lambda x: x % 2 == 0, data)
plus_five = map(lambda x: x + 5, evens)
total = reduce(lambda a, b: a + b, plus_five)

# 2. Nested List Comprehension (Flatten 2D matrix)
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6]
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Overcomplicating Lambdas:**
   - ❌ `lambda x: True if (x > 10 and x < 50 or x == 100) else False` (Unreadable spaghetti code).
   - ✅ Use a regular named `def` function for complex logic requiring multiple lines or conditions.

2. **Preferring `map()` + `lambda` over List Comprehensions:**
   - In modern Python, `[x * 2 for x in nums]` is generally considered cleaner and faster than `list(map(lambda x: x * 2, nums))`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the main difference between `map()` and `filter()`?
**Answer:** `map()` transforms every element in the input sequence to a new value (output list has the same length as input). `filter()` tests every element against a boolean predicate function and only keeps elements where the result is `True` (output list has length $\le$ input length).

### Q2: How does `reduce()` work with an optional `initializer` parameter?
**Answer:** `reduce(func, sequence, initializer)` puts `initializer` as the starting accumulator value before processing sequence elements. If the sequence is empty, `reduce` returns the `initializer` instead of raising a `TypeError`.

---

## 📝 Recap Checklist
- [x] Defined anonymous inline functions using `lambda`.
- [x] Applied `map()` for element-wise data transformations.
- [x] Applied `filter()` for boolean predicate data filtering.
- [x] Aggregated lists into single values using `functools.reduce()`.
- [x] Created List, Dictionary, and Set Comprehensions.
- [x] Processed real-world e-commerce, student, and employee datasets functionally.

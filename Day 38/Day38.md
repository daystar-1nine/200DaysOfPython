# 🐍 Day 38/200 – Masterclass Notes: Functional Programming in Python

🎯 **Goal:** Master **Functional Programming** paradigms in Python—understanding `lambda` anonymous functions, higher-order functions (`map()`, `filter()`, `functools.reduce()`), sorting with `key=lambda` parameters, boolean predicate aggregations (`any()`, `all()`), data pairing with `zip()`, item indexing with `enumerate()`, and knowing when functional expressions improve code vs. when explicit comprehensions or loops are cleaner.

---

## 📌 Executive Summary & Key Takeaways

- **First-Class Higher-Order Functions:** Functions in Python can accept other functions as arguments (`map`, `filter`, `sorted`) and return functions.
- **`lambda` Anonymous Functions:** Lightweight inline single-expression functions: `lambda x: x * 2`. Ideal for short key extractors or predicates.
- **`map(func, iterable)` & `filter(func, iterable)`:** Return lazy iterators applying transformation `func` to items or retaining items where `func(item)` is `True`.
- **`any()` and `all()`:** Boolean reduction short-circuiting:
  - `any(iterable)`: Returns `True` if at least one item evaluates to `True`.
  - `all(iterable)`: Returns `True` if every item evaluates to `True`.
- **`reduce(func, iterable)` vs. `sum()`:** `functools.reduce()` cumulatively combines elements. However, built-in aggregations like `sum()` or list comprehensions are often much clearer.

---

## 📖 Topic 1: Core Functional Tools (`map`, `filter`, `sorted`)

```python
from functools import reduce

numbers = [10, 15, 20, 25, 30]

# 1. Doubling numbers with map & lambda
doubled = list(map(lambda x: x * 2, numbers))

# 2. Filtering even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 3. Sorting by key in reverse
sorted_nums = sorted(numbers, key=lambda x: x, reverse=True)

# 4. Aggregating with reduce
total_sum = reduce(lambda acc, val: acc + val, numbers, 0)
```

---

## 📖 Topic 2: Data Validation & Inspection (`any`, `all`, `zip`, `enumerate`)

```python
expenses = [
    {"amount": 250, "category": "Food"},
    {"amount": 1200, "category": "Travel"},
    {"amount": 500, "category": "Shopping"}
]

# Check if any expense exceeds 1,000
has_large = any(e["amount"] > 1000 for e in expenses)

# Check if all expenses are positive
all_valid = all(e["amount"] > 0 for e in expenses)

# Indexing and pairing using enumerate & zip
categories = ["Food", "Travel", "Shopping"]
amounts = [250, 1200, 500]

for idx, (cat, amt) in enumerate(zip(categories, amounts), start=1):
    print(f"{idx}. {cat}: Rs.{amt}")
```

---

## ⚡ Master Cheat Sheet

```python
# Functional Python Quick Reference

# 1. Lambda syntax
add = lambda a, b: a + b

# 2. Functional Map & Filter
transformed = list(map(lambda x: x.upper(), strings))
filtered = list(filter(lambda x: len(x) > 3, strings))

# 3. Sorting objects by attribute/key
sorted_items = sorted(items, key=lambda item: item['score'], reverse=True)

# 4. Aggregation
has_admin = any(user.is_admin for user in users)
all_active = all(user.is_active for user in users)
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Overusing `lambda` and `reduce()` when Comprehensions are Clearer:**
   - ❌ `reduce(lambda a, b: a + b, [x['amount'] for x in items])` is hard to read.
   - ✅ `sum(x['amount'] for x in items)` is far more Pythonic and readable.

2. **Forgetting that `map()` and `filter()` Return Iterators:**
   - ❌ Trying to access `result[0]` directly on `map()` output throws `TypeError: 'map' object is not subscriptable`.
   - ✅ Convert to a `list` or iterate using a `for` loop.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: When should you use a list comprehension instead of `map()` with a `lambda`?
**Answer:** In Python, list comprehensions (e.g. `[x * 2 for x in numbers]`) are generally preferred over `map(lambda x: x * 2, numbers)` because they avoid lambda call overhead and are significantly clearer to read.

### Q2: How do `any()` and `all()` optimize performance during evaluation?
**Answer:** Both `any()` and `all()` perform short-circuit evaluation. `any()` stops and returns `True` immediately upon encountering the first truthy item; `all()` stops and returns `False` immediately upon encountering the first falsy item.

---

## 📝 Recap Checklist
- [x] Mastered `lambda`, `map()`, `filter()`, `reduce()`, and `sorted(key=...)`.
- [x] Used `any()` and `all()` for short-circuit validation predicates.
- [x] Used `zip()` and `enumerate()` for pairing and indexing.
- [x] Built the Functional Data Transformation Pipeline (`Day 38/pipeline/`).
- [x] Created Pytest test suite covering empty data, single/multiple records, sorting, filtering, and invalid inputs.

# 🐍 Day 34/200 – Masterclass Notes: Python Iterators & Iterables

🎯 **Goal:** Understand Python's underlying **Iteration Protocol**—distinguishing between **Iterables** and **Iterators**, implementing custom classes with `__iter__()` and `__next__()`, signaling completion with `StopIteration`, and building custom iterators for Countdowns, Even Numbers, Data Pagination, and Financial Transactions.

---

## 📌 Executive Summary & Key Takeaways

- **Iterable vs. Iterator:**
  - **Iterable:** An object (e.g. list, dict, str) that implements `__iter__()` to produce a fresh **Iterator**. Can be iterated over multiple times.
  - **Iterator:** A stateful object that implements both `__next__()` (to yield successive items or raise `StopIteration`) and `__iter__()` (returning `self`). It represents a single pass over a stream of data.
- **The Mechanics of `for` Loops:**
  ```python
  # What Python executes under the hood for: for x in iterable:
  iterator = iter(iterable)
  while True:
      try:
          x = next(iterator)
          # body of loop
      except StopIteration:
          break
  ```
- **State Management:** Custom iterators maintain internal pointer state (`self._index` or `self._current`) which increments on each call to `next()`.

---

## 📖 Topic 1: The Iterator Protocol in Practice

```python
class CountDown:
    def __init__(self, start: int):
        if start < 1:
            raise ValueError("Start number must be at least 1.")
        self.current = start

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration
        val = self.current
        self.current -= 1
        return val
```

---

## 📖 Topic 2: Pagination Iterator Pattern

```python
class PaginationIterator:
    def __init__(self, items: list, page_size: int = 2):
        if page_size < 1:
            raise ValueError("Page size must be at least 1.")
        self.items = items
        self.page_size = page_size
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> list:
        if self._index >= len(self.items):
            raise StopIteration
        page = self.items[self._index : self._index + self.page_size]
        self._index += self.page_size
        return page
```

---

## ⚡ Master Cheat Sheet

```python
# Iteration Protocol Quick Reference

# 1. Obtain Iterator from Iterable
it = iter([10, 20, 30])  # Calls [10, 20, 30].__iter__()

# 2. Advance Iterator
first = next(it)          # Returns 10
second = next(it)         # Returns 20

# 3. Custom Iterator Protocol Contract
class CustomIterator:
    def __iter__(self):
        return self       # An iterator MUST return self

    def __next__(self):
        if end_condition:
            raise StopIteration
        return next_value
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `return self` in `__iter__()`:**
   - ❌ Defining `__next__()` without `__iter__()` returning `self` makes the iterator un-usable in `for` loops.
   - ✅ Always implement `def __iter__(self): return self` on iterator classes.

2. **Re-using Exhausted Iterators:**
   - ❌ Iterators are single-pass streams. Once `StopIteration` is raised, further calls to `next()` continue raising `StopIteration`.
   - ✅ To re-iterate over a dataset, request a fresh iterator using `iter(iterable)`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: Is every Iterator an Iterable? Is every Iterable an Iterator?
**Answer:** Every Iterator is an Iterable because it implements `__iter__()` (returning `self`). However, not every Iterable is an Iterator; for example, a `list` is an Iterable, but calling `next(my_list)` will raise a `TypeError: 'list' object is not an iterator`.

### Q2: What happens when `next()` is called on an exhausted iterator?
**Answer:** The iterator raises `StopIteration`. The `for` loop catches this exception automatically and terminates cleanly.

---

## 📝 Recap Checklist
- [x] Mastered the difference between Iterables and Iterators.
- [x] Implemented `__iter__()` and `__next__()` protocols.
- [x] Signaled iteration completion using `raise StopIteration`.
- [x] Created `CountdownIterator`, `EvenNumberIterator`, `PaginationIterator`, and `TransactionIterator`.
- [x] Built Pytest suite testing first item, last item, StopIteration, empty iterators, and bounds checks.

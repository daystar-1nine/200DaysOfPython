# 🐍 Day 35/200 – Masterclass Notes: Generators, `yield` & Large File Processing

🎯 **Goal:** Master **Generators** and **Lazy Evaluation** in Python—understanding `yield`, generator functions, generator objects, generator expressions `(x for x in data)`, generator protocol methods (`send()`, `throw()`, `close()`), streaming data pipelines, and processing massive files (10,000+ to 1,000,000+ lines) with $O(1)$ constant memory usage instead of $O(N)$ memory crashes.

---

## 📌 Executive Summary & Key Takeaways

- **Lazy Evaluation vs. Eager Loading:**
  - **Eager Loading (`list`):** Computes and stores all items in RAM simultaneously. Memory usage scales linearly $O(N)$.
  - **Lazy Evaluation (`generator`):** Produces items one at a time on demand. Memory usage remains constant $O(1)$ regardless of dataset size (e.g. 100 lines vs. 1,000,000 lines).
- **The `yield` Keyword:** When a function contains `yield`, calling it returns a **Generator Object** without executing code immediately. Code executes up to `yield`, suspends state, returns the value, and resumes execution on the next call to `next()`.
- **Streaming Data Pipelines:** Generators can be chained together (`Reader` $\rightarrow$ `Filter` $\rightarrow$ `Transform` $\rightarrow$ `Output`) to build scalable, low-memory data processing pipelines.

---

## 📖 Topic 1: Generator Functions vs. Expressions

```python
# 1. Generator Function using yield
def fibonacci_generator(limit: int):
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

# 2. Generator Expression (Memory Efficient)
squares_gen = (x * x for x in range(1_000_000))
```

---

## 📖 Topic 2: Streaming File Reader Generator

```python
def read_lines(filename: str):
    """Streams file line-by-line using generator yield."""
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()
```

---

## ⚡ Master Cheat Sheet

```python
# Generators Quick Reference

# 1. Basic Generator Function
def simple_gen():
    yield "Item 1"
    yield "Item 2"

# 2. Memory Comparison
import sys
eager_list = [x for x in range(1_000_000)]
lazy_gen = (x for x in range(1_000_000))

print(sys.getsizeof(eager_list)) # ~8,000,000 bytes (8 MB)
print(sys.getsizeof(lazy_gen))   # ~200 bytes (Constant RAM)
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Calling `list(generator)` on Infinite Streams:**
   - ❌ `list(infinite_generator())` attempts to collect infinite values into memory, causing an Out-Of-Memory (OOM) system crash.
   - ✅ Consume generators using `for` loops, `next()`, or `itertools.islice()`.

2. **Re-using Exhausted Generator Objects:**
   - ❌ Generator objects are single-pass streams. Once fully yielded, subsequent iterations return nothing.
   - ✅ Call the generator function again to produce a new generator instance.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the main difference between `return` and `yield` in Python?
**Answer:** `return` terminates the function execution and returns a single value to the caller, discarding the function's stack frame. `yield` returns a value while preserving the function's local state and execution pointer, suspending the function until `next()` is called again.

### Q2: Why are generator pipelines preferred for big data processing?
**Answer:** Because each pipeline stage processes items lazily one-by-one. No intermediate datasets need to be written to disk or held in RAM, keeping overall memory footprint at $O(1)$.

---

## 📝 Recap Checklist
- [x] Understood `yield`, generator functions, and lazy evaluation.
- [x] Compared memory overhead between `list()` and generator expressions.
- [x] Built generator functions for Fibonacci, Even Numbers, and Squares.
- [x] Implemented line-by-line file reader generator (`reader.py`).
- [x] Built streaming data pipeline with filtering and transformations (`pipeline.py`).
- [x] Created Pytest test suite testing empty, small, large files, invalid lines, and filtering.

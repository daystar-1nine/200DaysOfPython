# 🐍 Day 14/200 – Masterclass Notes: Iterators & Generators

🎯 **Goal:** Master memory-efficient data processing in Python using the **Iterator Protocol** (`iter()`, `next()`, `StopIteration`), **Generator Functions** (`yield`), **Generator Expressions**, and **Lazy Evaluation**.

---

## 📌 Executive Summary & Key Takeaways

- **Iterable vs Iterator:**
  - **Iterable:** Any object that implements `__iter__()` or `__getitem__()` and can be passed to a `for` loop (e.g. lists, tuples, dicts, strings, sets).
  - **Iterator:** An object representing a stream of data that remembers its position and implements `__next__()`. Calling `next(iterator)` returns the next item or raises `StopIteration`.
- **Generators (`yield`):** Special functions that yield values one at a time on demand. When a generator encounters `yield`, it pauses execution, preserves its local variable state, and yields the value to the caller.
- **Lazy Evaluation:** Computing values on-demand as requested rather than loading an entire dataset into RAM at once.
- **Memory Efficiency:** Storing 1,000,000 integers in a Python list consumes ~8 MB of RAM; a generator producing 1,000,000 integers consumes only **112 bytes** of RAM regardless of sequence size!

---

## 📖 Topic 1: The Iterator Protocol (`iter()` & `next()`)

### 1.1 How `for` Loops Work Under the Hood

When you execute `for item in sequence:`, Python automatically calls:
1. `it = iter(sequence)` to obtain an iterator object.
2. `next(it)` repeatedly in a loop.
3. Catches `StopIteration` exception automatically to terminate the loop cleanly.

```python
numbers = [10, 20, 30]

# 1. Convert iterable to iterator
iterator = iter(numbers)

# 2. Retrieve items manually using next()
print(next(iterator))  # 10
print(next(iterator))  # 20
print(next(iterator))  # 30

# 3. Subsequent next() raises StopIteration
try:
    print(next(iterator))
except StopIteration:
    print("Reached end of iterator stream!")
```

---

## 📖 Topic 2: Generator Functions (`yield` vs `return`)

### 2.1 Comparison Matrix

| Feature | `return` Statement | `yield` Statement (Generator) |
|---|---|---|
| **Execution** | Terminates function execution completely | Pauses function execution and yields value |
| **State** | Destroys local variable state and stack frame | Preserves local variable state for next call |
| **Memory** | Builds and returns entire data structure in RAM | Streams items 1-by-1 (O(1) memory footprint) |
| **Return Type** | Single value or collection object | Returns a `generator` iterator object |

### 2.2 Generator Example
```python
def count_down(start):
    print("Beginning countdown...")
    while start > 0:
        yield start  # Pauses execution here and yields 'start'
        start -= 1

gen = count_down(3)
print(next(gen))  # Output: 3
print(next(gen))  # Output: 2
print(next(gen))  # Output: 1
```

---

## 📖 Topic 3: Generator Expressions

Generator expressions use tuple syntax `(expr for var in iterable)` and evaluate values lazily.

```python
# List Comprehension (Eager Evaluation - Allocates memory for 1,000,000 ints)
list_squares = [x * x for x in range(1000000)]  # ~8 MB RAM

# Generator Expression (Lazy Evaluation - 112 bytes RAM)
gen_squares = (x * x for x in range(1000000))   # 112 Bytes RAM!

print(next(gen_squares))  # 0
print(next(gen_squares))  # 1
```

---

## ⚡ Master Cheat Sheet

```python
# Iterators & Generators Cheat Sheet

# 1. Custom Class Iterator
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.current = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= self.limit:
            val = self.current
            self.current += 1
            return val
        raise StopIteration

# 2. Infinite Generator Stream
def infinite_ids():
    n = 1
    while True:
        yield f"ID-{n:05d}"
        n += 1

# 3. File Streaming Generator
def stream_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Exhausting Generators (One-Time Consumption):**
   - ❌ Generators can only be iterated **ONCE**. After a generator yields all its items, subsequent loops or `next()` calls will be empty / raise `StopIteration`.
   - ✅ If you need to iterate multiple times, re-instantiate the generator or convert to a list if memory allows (`list(gen)`).

2. **Accidentally Using `return` in Generators:**
   - In Python 3.3+, `return value` in a generator raises `StopIteration(value)` and terminates the generator.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is Lazy Evaluation and why is it beneficial in Data Engineering / ML pipelines?
**Answer:** Lazy evaluation means delaying the computation of values until they are explicitly needed by the consumer. In Data Engineering and Machine Learning, loading gigabytes of raw data into memory at once causes `MemoryError` crashes. Generators stream data line-by-line or batch-by-batch, keeping memory footprint minimal ($O(1)$ RAM).

### Q2: How does `yield from` work in nested generators?
**Answer:** `yield from subgenerator` delegates the generator operation to a sub-generator, allowing seamless composition of nested generator pipelines without writing explicit nested loops.

---

## 📝 Recap Checklist
- [x] Distinguished between Iterables and Iterators using `iter()` and `next()`.
- [x] Handled `StopIteration` exceptions manually and cleanly.
- [x] Wrote Generator Functions using `yield` and understood execution state pausing.
- [x] Used Generator Expressions for $O(1)$ memory-efficient calculations.
- [x] Built real-world streaming pipelines (file reader, step generator, Fibonacci stream).

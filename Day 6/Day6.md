# 🐍 Day 6/200 – Masterclass Notes: Python Lists, Operations & Comprehensions

🎯 **Goal:** Master Python lists, memory allocation behavior, positive/negative indexing, advanced slicing, list methods, iteration techniques, time complexity, and list comprehensions.

---

## 📌 Executive Summary & Key Takeaways

- **List (`list`):** An ordered, mutable, indexed collection that allows duplicate elements and mixed data types.
- **Indexing & Slicing:** Elements are accessed in $O(1)$ time via 0-based positive indexing or negative indexing (`-1` for last item). Slicing follows `[start:stop:step]`.
- **In-place vs Returning Methods:** Methods like `.sort()` and `.reverse()` mutate the list in-place and return `None`. Functions like `sorted()` and `reversed()` return new iterables.
- **List Comprehension:** Concise syntax for building lists: `[expr for item in iterable if condition]`.

---

## 📖 Topic 1: List Core Concepts & Memory Allocation

### 1.1 What is a List?
A list is a dynamic contiguous array of pointers pointing to Python objects in memory.

```python
fruits = ["Apple", "Banana", "Mango", "Orange"]
mixed = [42, "Hello", 3.14, True, [1, 2]]
```

### 1.2 Memory Allocation & Over-allocation Strategy
Python lists over-allocate extra slots when growing (via `.append()`) to achieve **$O(1)$ amortized time complexity** for insertions!

---

## 📖 Topic 2: Indexing & Advanced Slicing

### 2.1 Indexing
```python
fruits = ["Apple", "Banana", "Mango", "Orange"]

# Positive Indexing (0 to len-1)
print(fruits[0])   # Apple
print(fruits[2])   # Mango

# Negative Indexing (-1 to -len)
print(fruits[-1])  # Orange (Last item)
print(fruits[-2])  # Mango
```

### 2.2 Advanced Slicing Syntax (`list[start:stop:step]`)

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:6])    # [2, 3, 4, 5] (Index 2 up to 5)
print(numbers[:4])     # [0, 1, 2, 3] (First 4 items)
print(numbers[6:])     # [6, 7, 8, 9] (Index 6 to end)
print(numbers[::2])    # [0, 2, 4, 6, 8] (Every 2nd element)
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (Reversed list)
```

---

## 📖 Topic 3: Mutability & List Methods

### 3.1 Adding Elements

| Method | Description | Time Complexity | Example |
|---|---|:---:|---|
| `append(x)` | Adds `x` to the end | $O(1)$ Amortized | `lst.append(5)` |
| `insert(i, x)` | Inserts `x` at index `i` | $O(n)$ | `lst.insert(0, "First")` |
| `extend(iterable)` | Appends all items from iterable | $O(k)$ | `lst.extend([10, 20])` |

### 3.2 Removing Elements

| Method | Description | Error Handling | Example |
|---|---|---|---|
| `remove(x)` | Removes first occurrence of value `x` | ❌ `ValueError` if `x` missing | `lst.remove("Apple")` |
| `pop([i])` | Removes & returns item at index `i` (default last) | ❌ `IndexError` if empty/invalid | `item = lst.pop(0)` |
| `del lst[i]` | Deletes element at index `i` or slice | ❌ `IndexError` if out of bounds | `del lst[0]` |
| `clear()` | Empties all elements from list | Safe | `lst.clear()` |

### 3.3 Sorting and Reversing ⚠️
```python
nums = [5, 2, 8, 1, 9]

# In-place sorting (Mutates original, returns None!)
nums.sort()
print(nums)  # [1, 2, 5, 8, 9]

# Reverse sorting in-place
nums.sort(reverse=True)
print(nums)  # [9, 8, 5, 2, 1]

# Functional sorting (Leaves original list unchanged, returns new list)
original = [3, 1, 4]
new_sorted = sorted(original)
print(original)   # [3, 1, 4]
print(new_sorted) # [1, 3, 4]
```

---

## 📖 Topic 4: List Comprehension Deep Dive

Syntax: `[expression for item in iterable if condition]`

```python
# 1. Basic Transformation: Square numbers 0-9
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 2. Filtering: Keep only even numbers
evens = [x for x in range(20) if x % 2 == 0]

# 3. If-Else inside List Comprehension:
# Format: [val_if_true if condition else val_if_false for item in iterable]
labels = ["Even" if x % 2 == 0 else "Odd" for x in range(5)]
print(labels)  # ['Even', 'Odd', 'Even', 'Odd', 'Even']
```

---

## ⚡ Master Cheat Sheet & Time Complexity Matrix

| Operation | Method / Syntax | Time Complexity |
|---|---|:---:|
| Access Element | `lst[i]` | $O(1)$ |
| Append Element | `lst.append(x)` | $O(1)$ Amortized |
| Insert Element | `lst.insert(0, x)` | $O(n)$ |
| Delete Element | `del lst[i]` / `lst.pop(0)` | $O(n)$ |
| Search Element | `x in lst` | $O(n)$ |
| Slicing | `lst[a:b]` | $O(k)$ where $k=b-a$ |
| Sort List | `lst.sort()` | $O(n \log n)$ (Timsort) |

---

## ⚠️ Common Pitfalls & Best Practices

1. **Confusing `.sort()` and `sorted()`:**
   - `.sort()` returns `None`! `my_list = my_list.sort()` sets `my_list` to `None`!

2. **Shallow Copy vs Deep Copy:**
   - `list2 = list1` creates a reference alias, NOT a copy! Modifying `list2` modifies `list1`.
   - Use `list2 = list1.copy()` or `list2 = list1[:]` for a shallow copy.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the time complexity of searching for an item in a list vs a set?
**Answer:** Searching in a list (`x in my_list`) takes $O(n)$ linear time. Searching in a set (`x in my_set`) takes $O(1)$ average hash lookup time.

### Q2: How do you reverse a list in Python using slicing?
**Answer:** `reversed_list = original_list[::-1]`

### Q3: What happens when you assign `a = [1, 2]` and `b = a`?
**Answer:** Both `a` and `b` point to the exact same list instance in memory. Any modification via `b.append(3)` will be reflected in `a`.

---

## 📝 Recap Checklist
- [x] Understood zero-based positive & negative list indexing.
- [x] Mastered slicing syntax `[start:stop:step]`.
- [x] Differentiated in-place `.sort()` from functional `sorted()`.
- [x] Mastered list comprehensions with filtering and conditional expressions.

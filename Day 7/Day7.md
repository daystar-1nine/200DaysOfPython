# 🐍 Day 7/200 – Masterclass Notes: Tuples, Sets & Dictionaries

🎯 **Goal:** Master Python's three primary collection data types (Tuples, Sets, and Dictionaries) from absolute fundamentals to advanced internal mechanics, memory optimization, and interview-level depth.

---

## 📌 Quick Navigation & Collection Matrix

### 🚀 Comparison Matrix at a Glance

| Feature | List (`list`) | Tuple (`tuple`) | Set (`set`) | Dictionary (`dict`) |
|---|:---:|:---:|:---:|:---:|
| **Syntax** | `[1, 2, 3]` | `(1, 2, 3)` | `{1, 2, 3}` | `{"a": 1, "b": 2}` |
| **Ordered** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes (Insertion Order Python 3.7+) |
| **Mutable** | ✅ Yes | ❌ No (Immutable) | ✅ Yes | ✅ Yes (Keys Immutable) |
| **Duplicates** | ✅ Allowed | ✅ Allowed | ❌ Not Allowed | Keys: ❌ / Values: ✅ |
| **Indexing** | Positive & Negative | Positive & Negative | ❌ Not Supported | By Key `dict["key"]` |
| **Lookup Time** | $O(n)$ Linear | $O(n)$ Linear | $O(1)$ Average Hash | $O(1)$ Average Hash |
| **Memory Overhead** | Higher (Over-allocates) | Lowest (Fixed size) | High (Hash Table) | High (Hash Table) |

---

## 📖 Topic 1: Tuples — The Immutable Workhorse

### 1.1 What is a Tuple?
A **tuple** is an ordered, immutable collection of Python objects. Once created, its length and elements cannot be added, removed, or replaced.

```python
# Creating a simple tuple
coordinates = (19.0760, 72.8777)
rgb_color = (255, 128, 0)
```

---

### 1.2 Syntax Quirks & Single-Element Tuples ⚠️
A common beginner trap is creating a single-element tuple. Parentheses alone **do not** make a tuple; the **comma** does!

```python
# ❌ INCORRECT: Python interprets this as an integer in parentheses!
not_a_tuple = (5)
print(type(not_a_tuple))  # <class 'int'>

# ✅ CORRECT: The trailing comma defines a single-element tuple
is_a_tuple = (5,)
print(type(is_a_tuple))   # <class 'tuple'>

# Parentheses are optional when defining tuples (Tuple Packing)
point = 10, 20
print(type(point))        # <class 'tuple'>
```

---

### 1.3 Tuple Unpacking & Extended Unpacking
Tuples allow clean variable assignment in a single line.

```python
# Basic Unpacking
point = (10, 20, 30)
x, y, z = point
print(f"x={x}, y={y}, z={z}")

# Swapping two variables without a temp variable
a, b = 5, 10
a, b = b, a  # Uses tuple packing/unpacking behind the scenes!

# Extended Unpacking using asterisk (*)
numbers = (1, 2, 3, 4, 5, 6)
first, *middle, last = numbers
print(first)   # 1
print(middle)  # [2, 3, 4, 5] (Unpacked as a list)
print(last)    # 6
```

---

### 1.4 Why use Tuples over Lists? (Internal Mechanics & Performance)

1. **Memory Efficiency:** Tuples use less memory than lists because they are fixed-size and do not allocate extra buffer capacity for future insertions.
2. **Speed:** Accessing and iterating over tuples is slightly faster than lists.
3. **Data Integrity:** Protects data against unintended modifications (Read-only data safety).
4. **Dictionary Keys:** Tuples can be used as keys in dictionaries (lists cannot because lists are mutable/unhashable).

```python
import sys

list_demo = [1, 2, 3, 4, 5]
tuple_demo = (1, 2, 3, 4, 5)

print("List Size in Bytes:", sys.getsizeof(list_demo))   # ~104 bytes
print("Tuple Size in Bytes:", sys.getsizeof(tuple_demo)) # ~80 bytes
```

---

### 1.5 Immutability Edge Case: Tuples containing Mutable Objects
> 🧠 **Crucial Insight:** A tuple's *references* are immutable, but the *referenced objects* themselves can still be mutable!

```python
# A tuple containing a list
mixed_tuple = (1, 2, [10, 20])

# ❌ You CANNOT replace the list reference itself:
# mixed_tuple[2] = [30, 40]  # Raises TypeError!

# ✅ You CAN modify the contents of the internal list:
mixed_tuple[2].append(30)
print(mixed_tuple)  # Output: (1, 2, [10, 20, 30])
```

---

### 1.6 Useful Tuple Methods
Since tuples are immutable, they only have 2 built-in methods:

| Method | Description | Example |
|---|---|---|
| `count(x)` | Counts occurrences of `x` | `(1, 2, 2, 3).count(2) -> 2` |
| `index(x)` | Returns first index of `x` | `("a", "b", "c").index("b") -> 1` |

---

### 1.7 Named Tuples (`collections.namedtuple`)
For readable code, `namedtuple` from the `collections` module allows accessing elements by field name in addition to index:

```python
from collections import namedtuple

# Define a named tuple structure
Point = namedtuple("Point", ["x", "y"])
p1 = Point(x=10, y=20)

print(p1.x)  # Access by field name -> 10
print(p1[1]) # Access by index -> 20
```

---

## 📖 Topic 2: Sets — The Unique & Unordered Collection

### 2.1 What is a Set?
A **set** is an unordered collection of unique, hashable items. Duplicate values are automatically discarded upon insertion.

```python
# Creating a set
unique_numbers = {1, 2, 3, 3, 2, 4}
print(unique_numbers)  # Output: {1, 2, 3, 4}

# ⚠️ Creating an empty set MUST use set(), NOT {}
empty_dict = {}     # <class 'dict'>
empty_set = set()   # <class 'set'>
```

---

### 2.2 Mathematical Set Operations
Sets in Python support full mathematical set theory operations:

```python
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

# 1. Union (All unique items from both sets)
print(A | B)             # {1, 2, 3, 4, 5, 6}
print(A.union(B))

# 2. Intersection (Items present in BOTH sets)
print(A & B)             # {3, 4}
print(A.intersection(B))

# 3. Difference (Items in A but NOT in B)
print(A - B)             # {1, 2}
print(A.difference(B))

# 4. Symmetric Difference (Items in A or B, but NOT in both)
print(A ^ B)             # {1, 2, 5, 6}
print(A.symmetric_difference(B))
```

---

### 2.3 Set Methods: `remove()` vs `discard()` ⚠️

| Method | Behavior if item exists | Behavior if item missing |
|---|---|---|
| `add(elem)` | Adds element | N/A |
| `remove(elem)` | Removes element | ❌ Raises `KeyError` |
| `discard(elem)` | Removes element | ✅ Does nothing (Safe!) |
| `pop()` | Removes & returns arbitrary element | ❌ Raises `KeyError` if set empty |
| `clear()` | Empties the set | ✅ Set becomes empty `set()` |

```python
s = {10, 20, 30}

s.discard(99) # Safe! No error raised.
# s.remove(99) # ❌ KeyError: 99
```

---

### 2.4 Time Complexity & Hash Tables
Sets are implemented using **Hash Tables**.
- Checking membership (`item in my_set`) takes **$O(1)$ average time complexity**, compared to $O(n)$ in a list!

```python
import time

large_list = list(range(10_000_000))
large_set = set(large_list)

# Checking membership in List (O(n))
start = time.time()
9_999_999 in large_list
print(f"List Lookup Time: {time.time() - start:.6f} seconds")

# Checking membership in Set (O(1))
start = time.time()
9_999_999 in large_set
print(f"Set Lookup Time: {time.time() - start:.6f} seconds")  # Extremely fast!
```

---

### 2.5 Frozenset (Immutable Set)
If you need an immutable set (for example, to use a set as a dictionary key or inside another set), use `frozenset`:

```python
fs = frozenset([1, 2, 3, 4])
# fs.add(5)  # AttributeError: 'frozenset' object has no attribute 'add'

dict_with_set_key = {fs: "Allowed!"}
print(dict_with_set_key[fs])  # Output: Allowed!
```

---

## 📖 Topic 3: Dictionaries — Key-Value Powerhouse

### 3.1 What is a Dictionary?
A **dictionary** stores data in `key: value` pairs. Keys must be unique and **hashable** (immutable types like strings, numbers, or tuples).

```python
user = {
    "username": "suraj_99",
    "email": "surajonenine@gmail.com",
    "is_active": True,
    "role": "Admin"
}
```

---

### 3.2 Key Dictionary Operations & Safe Access
Accessing a missing key using bracket notation `dict["missing_key"]` raises a `KeyError`. Always use `.get()` or `setdefault()` for safe access!

```python
profile = {"name": "Suraj", "age": 20}

# ❌ Risky: Raises KeyError if key missing
# print(profile["city"])

# ✅ Safe Access using .get(key, default_value)
print(profile.get("city", "Mumbai"))  # Output: Mumbai (Fallback used)

# setdefault(key, default) returns value if present, else inserts key with default
city = profile.setdefault("city", "Pune")
print(profile)  # Output: {'name': 'Suraj', 'age': 20, 'city': 'Pune'}
```

---

### 3.3 Iterating Through Dictionaries Efficiently

```python
student = {"name": "Rahul", "marks": 92, "grade": "A+"}

# 1. Iterate over Keys
for key in student:
    print(key)

# 2. Iterate over Values
for value in student.values():
    print(value)

# 3. Iterate over Key-Value pairs (Recommended)
for key, value in student.items():
    print(f"{key} -> {value}")
```

---

### 3.4 Dictionary Merging & Comprehensions

#### Merging Dictionaries (Python 3.9+)
Python 3.9 introduced the union operator `|` for dictionaries:

```python
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 99, "c": 3}

# Merging with | (Right side overrides left side for duplicates)
merged = dict1 | dict2
print(merged)  # Output: {'a': 1, 'b': 99, 'c': 3}

# Unpacking method (Pre-Python 3.9)
merged_old = {**dict1, **dict2}
```

#### Dictionary Comprehension Syntax
`{key_expr: value_expr for item in iterable if condition}`

```python
# Creating a dictionary of number squares for even numbers
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
print(even_squares)  # Output: {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}
```

---

### 3.5 Useful Modules: `collections` (`defaultdict`, `Counter`)

```python
from collections import defaultdict, Counter

# 1. defaultdict avoids KeyErrors when grouping items
word_groups = defaultdict(list)
words = ["apple", "banana", "apricot", "berry", "cherry"]

for w in words:
    word_groups[w[0]].append(w)  # Group by first letter

print(dict(word_groups))
# Output: {'a': ['apple', 'apricot'], 'b': ['banana', 'berry'], 'c': ['cherry']}

# 2. Counter counts element frequencies automatically
text = "abracadabra"
counts = Counter(text)
print(counts)  # Output: Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
```

---

## ⚡ Master Cheat Sheet & Decision Flowchart

```text
Do you need to store data in key-value pairs?
├── YES ──> Use DICTIONARY (dict)
└── NO
    ├── Do you need to prevent duplicate values?
    │   └── YES ──> Use SET (set)
    └── NO
        ├── Will the elements change after creation?
        │   ├── YES ──> Use LIST (list)
        │   └── NO  ──> Use TUPLE (tuple)
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Mutable Objects as Dictionary Keys:**
   - ❌ Never use a `list` or `dict` as a key in a dictionary because they are mutable/unhashable.
   - ✅ Use a `tuple` or `str` instead.

2. **Modifying a Collection While Iterating:**
   - ❌ Removing items from a list/set/dict while looping over it causes bugs or runtime errors.
   - ✅ Iterate over a copy (`for item in list(my_collection):`) or use comprehension.

3. **Confusing `{}` with `set()`:**
   - ❌ `{}` creates an empty dictionary, NOT a set.
   - ✅ Use `set()` for an empty set.

---

## 💡 Memory Tricks & Mnemonics

- **TUPLE:** **T**ight & **U**nchangeable (**T**uple = **T**uned for speed & immutability).
- **SET:** **S**ingular & **E**lementary (**S**et = **S**ingular/unique items only).
- **DICT:** **D**irect **I**ndexed **C**atalog **T**able (**D**ict = Lookup by Key).

---

## ❓ Interview & Practice Questions (With Solutions)

### Q1: How do you swap two variables without using a 3rd temporary variable in Python?
**Answer:** Use tuple unpacking:
```python
a, b = 10, 20
a, b = b, a
print(a, b)  # 20, 10
```

### Q2: What is the difference between `dict.get("key")` and `dict["key"]`?
**Answer:** `dict["key"]` raises a `KeyError` if `"key"` does not exist in the dictionary. `dict.get("key")` safely returns `None` (or a specified default fallback value) without crashing the program.

### Q3: How do you merge two dictionaries in Python 3.9+?
**Answer:** Use the dictionary union operator `|`:
```python
dict1 = {"x": 1}
dict2 = {"y": 2}
merged = dict1 | dict2
```

### Q4: Can a tuple be modified if it contains a list inside it?
**Answer:** The tuple itself cannot have elements added, removed, or reassigned. However, the internal list object stored within the tuple *can* be modified in-place (e.g. using `.append()`).

---

## 📝 Recap Checklist
- [x] Know when to pick `tuple`, `set`, or `dict`.
- [x] Understand set operations (`union`, `intersection`, `difference`).
- [x] Master dictionary `.get()`, `.items()`, and comprehension syntax.
- [x] Remember that sets provide $O(1)$ membership lookups.

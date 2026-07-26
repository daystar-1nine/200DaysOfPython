# 🐍 Day 7/200 – Tuples, Sets & Dictionaries

🎯 **Goal:** Learn the differences between Python's three major collection data types (Tuples, Sets, Dictionaries) and understand when to use each one.

---

## 📚 Topic 1 – Tuples
A **tuple** is an ordered, immutable (unchangeable) collection of elements. Tuples are defined using round brackets `()`.

```python
fruits = ("Apple", "Banana", "Mango")
print(fruits)

# Accessing elements (Indexing)
print(fruits[0])   # Output: Apple
print(fruits[-1])  # Output: Mango
```

### Useful Tuple Methods
- `count(x)`: Returns the number of times `x` appears in the tuple.
- `index(x)`: Returns the index of the first occurrence of `x`.
- `len(tuple)`: Returns the length of the tuple.

```python
numbers = (10, 20, 30, 20)

print(numbers.count(20)) # Output: 2
print(numbers.index(30)) # Output: 2
print(len(numbers))      # Output: 4
```

### When to use Tuples?
- **Fixed / Immutable Data:** Geographic coordinates `(latitude, longitude)`.
- **Days of the week or months:** `("Mon", "Tue", "Wed", ...)`
- **RGB Colors:** `(255, 0, 128)`
- **Database Records:** Returning fixed rows of data.

---

## 📚 Topic 2 – Sets
A **set** is an unordered collection of **unique** elements. Sets are defined using curly braces `{}`. Duplicate elements are automatically ignored.

```python
numbers = {1, 2, 3, 2, 1, 4}
print(numbers)  # Output: {1, 2, 3, 4}
```

### Adding & Removing Items
```python
fruits = {"Apple", "Banana"}

# Adding items
fruits.add("Mango")

# Removing items
fruits.remove("Apple")  # Raises KeyError if item doesn't exist
fruits.discard("Kiwi")  # Does NOT raise an error if item is missing
```

### Set Operations

| Operation | Method / Syntax | Description | Example |
|---|---|---|---|
| **Union** | `A.union(B)` or `A \| B` | Combines elements from both sets | `{1, 2} \| {2, 3} -> {1, 2, 3}` |
| **Intersection** | `A.intersection(B)` or `A & B` | Elements common to both sets | `{1, 2} & {2, 3} -> {2}` |
| **Difference** | `A.difference(B)` or `A - B` | Elements in A but not in B | `{1, 2} - {2, 3} -> {1}` |

```python
A = {1, 2, 3}
B = {3, 4, 5}

print(A.union(B))        # Output: {1, 2, 3, 4, 5}
print(A.intersection(B)) # Output: {3}
print(A.difference(B))   # Output: {1, 2}
```

### When to use Sets?
- **Removing duplicates** from a list.
- **Fast membership checking** (`item in my_set` is $O(1)$).
- **Finding common items or unique tags**.

---

## 📚 Topic 3 – Dictionaries
A **dictionary** is an ordered (since Python 3.7+), mutable collection of **key-value pairs**. Keys must be unique and immutable.

```python
student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.85
}

# Accessing values
print(student["name"])     # Output: Suraj
print(student.get("cgpa")) # Output: 8.85

# Updating value
student["age"] = 21

# Adding new key-value pair
student["city"] = "Mumbai"

# Removing key-value pair
del student["city"]
```

### Useful Dictionary Methods
- `keys()`: Returns all keys.
- `values()`: Returns all values.
- `items()`: Returns all (key, value) tuple pairs.
- `get(key, default)`: Safely returns value for key without raising KeyError.
- `pop(key)`: Removes specified key and returns value.
- `update(dict2)`: Updates dictionary with another dictionary.

```python
for key, value in student.items():
    print(f"{key} -> {value}")
```

---

## 🔥 Difference Between Collections

| Feature | List | Tuple | Set | Dictionary |
|---|:---:|:---:|:---:|:---:|
| **Ordered** | ✅ | ✅ | ❌ | ✅ |
| **Mutable** | ✅ | ❌ | ✅ | ✅ |
| **Duplicates** | ✅ Allowed | ✅ Allowed | ❌ No | Keys ❌ / Values ✅ |
| **Indexing** | By Index `[0]` | By Index `[0]` | ❌ No Index | By Key `["name"]` |
| **Syntax** | `[1, 2, 3]` | `(1, 2, 3)` | `{1, 2, 3}` | `{"a": 1, "b": 2}` |

---

## 💡 Summary
- Use **Lists** when you need an ordered sequence that will change.
- Use **Tuples** for read-only / fixed data that shouldn't change.
- Use **Sets** when uniqueness matters or set math operations are needed.
- Use **Dictionaries** when mapping descriptive keys to values (real-world objects, records).

# 🐍 Day 32/200 – Masterclass Notes: Dunder / Magic Methods & Smart Collection

🎯 **Goal:** Understand how Python's built-in protocols and syntax (e.g. `len()`, `[]`, `in`, `+`, `==`, `for in`) delegate directly to special **Dunder (Double Underscore) Methods**—mastering object representation (`__str__`, `__repr__`), sequence container emulation (`__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`), iteration (`__iter__`), comparison (`__eq__`, `__lt__`), and arithmetic operator overloading (`__add__`, `__sub__`).

---

## 📌 Executive Summary & Key Takeaways

- **Dunder Methods Protocol:** Python uses dunder methods under the hood whenever built-in operators or functions are executed on an object (e.g. `len(obj)` calls `obj.__len__()`, `x in obj` calls `obj.__contains__(x)`).
- **`__str__` vs. `__repr__`:**
  - **`__str__`:** Returns user-friendly, clean string representation meant for `print()`.
  - **`__repr__`:** Returns developer-friendly, unambiguous representation meant for debugging (ideally executable Python code like `SmartCollection(['A', 'B'])`).
- **Container Emulation (`Sequence` protocol):** Implementing `__len__`, `__getitem__`, `__setitem__`, `__delitem__`, and `__contains__` allows custom objects to act seamlessly like built-in lists or tuples.
- **Operator Overloading:** Implementing `__add__`, `__sub__`, `__eq__`, `__lt__` allows custom objects to support mathematical operators (`+`, `-`) and comparisons (`==`, `<`).

---

## 📖 Topic 1: String Representation Protocols

```python
class Book:
    def __init__(self, title: str, author: str, pages: int):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self) -> str:
        """User-friendly representation for print(book)."""
        return f"'{self.title}' by {self.author}"

    def __repr__(self) -> str:
        """Developer-friendly representation for repr(book)."""
        return f"Book(title={self.title!r}, author={self.author!r}, pages={self.pages})"
```

---

## 📖 Topic 2: Container Protocol & Indexing

```python
class SmartCollection:
    def __init__(self, items: list | None = None):
        self._items = list(items) if items else []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice):
        return self._items[index]

    def __contains__(self, item) -> bool:
        return item in self._items
```

---

## 📖 Topic 3: Operator Overloading

```python
class Money:
    def __init__(self, amount: float):
        self.amount = amount

    def __add__(self, other: "Money") -> "Money":
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount + other.amount)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount
```

---

## ⚡ Master Cheat Sheet

```python
# Dunder Methods Quick Reference

# Built-in Syntax        --> Target Dunder Method
# print(obj), str(obj)   --> obj.__str__()
# repr(obj)              --> obj.__repr__()
# len(obj)               --> obj.__len__()
# obj[i]                 --> obj.__getitem__(i)
# obj[i] = val           --> obj.__setitem__(i, val)
# del obj[i]             --> obj.__delitem__(i)
# item in obj            --> obj.__contains__(item)
# for item in obj:       --> obj.__iter__()
# obj1 == obj2           --> obj1.__eq__(obj2)
# obj1 + obj2            --> obj1.__add__(obj2)
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Returning Non-String Values in `__str__` or `__repr__`:**
   - ❌ `def __str__(self): return self.amount` (raises `TypeError`).
   - ✅ Always return a `str` type.

2. **Not Handling Unrecognized Types in Operator Overloading:**
   - ❌ `def __add__(self, other): return Money(self.amount + other)` (fails if `other` is not Money).
   - ✅ Check `if not isinstance(other, Money): return NotImplemented`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the fallback behavior if an object implements `__getitem__` but not `__iter__`?
**Answer:** Python's iteration protocol will fall back to using `__getitem__` starting from index `0` and incrementing until `IndexError` is raised.

### Q2: What happens if `__eq__` returns `NotImplemented`?
**Answer:** Python will attempt to call the reverse comparison method `other.__eq__(self)`. If both return `NotImplemented`, Python falls back to comparing object memory identities (`is`).

---

## 📝 Recap Checklist
- [x] Implemented `__str__` and `__repr__` for custom objects.
- [x] Implemented container sequence methods (`__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`).
- [x] Overloaded arithmetic operators (`__add__`, `__sub__`).
- [x] Overloaded comparison operators (`__eq__`, `__lt__`).
- [x] Built the `SmartCollection` class supporting indexing, iteration, slicing, and addition.
- [x] Created full Pytest test suite covering 10+ edge cases.

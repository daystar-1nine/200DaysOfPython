# 🐍 Day 22/200 – Masterclass Notes: Type Hints & Dataclasses

🎯 **Goal:** Master static type annotations (**Type Hints**), collection generics (`list`, `dict`, `tuple`, `set`), Union/Optional types (`T | None`), `TypedDict` schemas, and Python's **`dataclasses`** module for writing clean, self-documenting, and maintainable software.

---

## 📌 Executive Summary & Key Takeaways

- **Type Hints:** Annotations added to variables, function parameters, and return types.
  - Basic: `name: str = "Suraj"`, `age: int = 20`
  - Function: `def add(a: int, b: int) -> int:`
- **Generics & Built-in Containers:**
  - Modern Python 3.9+ uses built-in collections directly: `list[str]`, `dict[str, int]`, `tuple[str, int]`, `set[int]`.
- **Optional & Union Types:**
  - `str | None` represents a value that is either a string or `None`.
  - `int | float` accepts either integer or floating-point numbers.
- **TypedDict (`from typing import TypedDict`):** Defines fixed keys and value types for standard Python dictionaries.
- **Dataclasses (`from dataclasses import dataclass`):**
  - Automatically generates `__init__()`, `__repr__()`, `__eq__()`, and helper methods for classes primarily designed to hold data.
  - Supports default values, factory defaults (`field(default_factory=list)`), methods, and immutability (`frozen=True`).

---

## 📖 Topic 1: Type Hints & Function Annotations

### 1.1 Collections & Function Return Types

```python
# Collections
user_list: list[str] = ["Suraj", "Rahul", "Priya"]
user_scores: dict[str, float] = {"Suraj": 95.5, "Rahul": 88.0}
user_pair: tuple[str, int] = ("Suraj", 101)

# Function Annotations
def calculate_average(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

# Optional & Union Types
def get_user_by_id(user_id: int) -> str | None:
    db = {1: "Suraj", 2: "Rahul"}
    return db.get(user_id)
```

---

## 📖 Topic 2: TypedDict Schema

```python
from typing import TypedDict

# Defines fixed key structure for dictionaries
class UserProfile(TypedDict):
    id: int
    username: str
    email: str
    is_active: bool

# Usage
profile: UserProfile = {
    "id": 101,
    "username": "suraj19",
    "email": "suraj@example.com",
    "is_active": True
}
```

---

## 📖 Topic 3: Python Dataclasses (`@dataclass`)

### 3.1 Basic Dataclass vs Standard OOP

```python
from dataclasses import dataclass, field

@dataclass
class Product:
    id: int
    name: str
    price: float
    tags: list[str] = field(default_factory=list)  # Mutable default factory

    def total_value(self, quantity: int) -> float:
        return self.price * quantity

# Automatically generates __init__(id, name, price, tags) and __repr__
laptop = Product(id=1, name="Laptop", price=65000.0)
laptop.tags.append("Electronics")
print(laptop)  # Product(id=1, name='Laptop', price=65000.0, tags=['Electronics'])
```

### 3.2 Frozen (Immutable) Dataclasses

```python
@dataclass(frozen=True)
class Point:
    x: int
    y: int

p = Point(10, 20)
# p.x = 30  # Raises FrozenInstanceError! Immutable object.
```

---

## ⚡ Master Cheat Sheet

```python
# Type Hints & Dataclasses Cheat Sheet

from dataclasses import dataclass, field
from typing import TypedDict

# 1. Type Hints
def process_data(items: list[int], factor: float = 1.0) -> list[float]:
    return [x * factor for x in items]

# 2. TypedDict
class ConfigDict(TypedDict):
    host: str
    port: int

# 3. Dataclass with Factory & Frozen
@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int = 8080
    endpoints: tuple[str, ...] = field(default_factory=tuple)
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Mutable Default Argument in Dataclasses:**
   - ❌ `skills: list[str] = []` (Raises `ValueError: mutable default [] is not allowed`).
   - ✅ Always use `skills: list[str] = field(default_factory=list)`.

2. **Expecting Type Hints to Enforce Types at Runtime Automatically:**
   - Python type hints are purely annotations. Python does NOT raise runtime errors if wrong types are passed unless validated with static type checkers like `mypy` or libraries like `pydantic`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the main advantage of `@dataclass` over a normal Python class?
**Answer:** `@dataclass` automatically generates boilerplate methods (`__init__`, `__repr__`, `__eq__`, `__hash__`) based on field type annotations, drastically reducing boilerplate code for data container objects.

### Q2: Why should you use `field(default_factory=list)` instead of `skills = []` in a dataclass?
**Answer:** Setting a mutable object (like a list or dict) directly as a default value shares that single object instance across all instances of the class. `default_factory=list` invokes a fresh list constructor for every new instance created.

---

## 📝 Recap Checklist
- [x] Applied primitive and collection type hints (`list[T]`, `dict[K,V]`, `tuple`, `set`).
- [x] Defined optional/union types using modern `T | None` syntax.
- [x] Structured dict payloads using `typing.TypedDict`.
- [x] Implemented dataclasses using `@dataclass`, `field(default_factory=...)`, and `frozen=True`.
- [x] Integrated dataclasses with API JSON payloads.
- [x] Built Student Management System, Inventory System, and API Data Model Converter projects.

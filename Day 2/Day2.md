# 🐍 Day 2/200 – Masterclass Notes: Variables, Operators & Type Conversion

🎯 **Goal:** Master Python operators (arithmetic, comparison, logical, assignment, bitwise, identity, membership), implicit/explicit type conversion, precedence rules, and mathematical expressions.

---

## 📌 Executive Summary & Operator Quick Reference

- **Data Types:** `int`, `float`, `str`, `bool`.
- **Implicit Conversion:** Python automatically upcasts narrower types to wider types (e.g. `int + float -> float`).
- **Explicit Conversion (Type Casting):** Manually converting using `int()`, `float()`, `str()`, `bool()`.
- **Operators:** Arithmetic (`+`, `-`, `*`, `/`, `//`, `%`, `**`), Comparison (`==`, `!=`, `>`, `<`, `>=`, `<=`), Logical (`and`, `or`, `not`), Identity (`is`, `is not`), Membership (`in`, `not in`).

---

## 📖 Topic 1: Operators Deep Dive

### 1.1 Arithmetic Operators

| Operator | Name | Example | Description |
|---|---|---|---|
| `+` | Addition | `10 + 5 -> 15` | Adds two numbers |
| `-` | Subtraction | `10 - 5 -> 5` | Subtracts second number from first |
| `*` | Multiplication | `10 * 5 -> 50` | Multiplies two numbers |
| `/` | Division | `10 / 4 -> 2.5` | **Always returns a float** |
| `//` | Floor Division | `10 // 4 -> 2` | Divides and rounds down to nearest integer |
| `%` | Modulo | `10 % 3 -> 1` | Returns remainder of division |
| `**` | Exponentiation | `2 ** 3 -> 8` | Power operation ($2^3$) |

> ⚠️ **Key Trap (`/` vs `//`):**
> Standard division `/` **always** results in a `float`, even if the numbers divide evenly! `4 / 2 -> 2.0`.
> Floor division `//` rounds down towards negative infinity (`-7 // 2 -> -4`).

---

### 1.2 Comparison Operators
Comparison operators evaluate expressions and return a boolean result (`True` or `False`).

```python
x, y = 10, 20

print(x == y)  # False (Equal to)
print(x != y)  # True  (Not equal to)
print(x > y)   # False (Greater than)
print(x < y)   # True  (Less than)
print(x >= 10) # True  (Greater than or equal to)
print(y <= 20) # True  (Less than or equal to)
```

---

### 1.3 Logical Operators (`and`, `or`, `not`)

| Logical Operator | Short-Circuit Behavior | Evaluates True When |
|---|---|---|
| `and` | Stops if **first** operand is `False` | BOTH operands are `True` |
| `or` | Stops if **first** operand is `True` | AT LEAST ONE operand is `True` |
| `not` | Flips boolean truth value | Operand is `False` |

```python
age = 22
has_license = True

# Logical AND
can_drive = age >= 18 and has_license  # True

# Short-Circuit Evaluation Example:
# Python skips evaluating (10 / 0) because 5 > 10 is already False!
result = (5 > 10) and (10 / 0 == 1)    # Output: False (No ZeroDivisionError!)
```

---

### 1.4 Identity (`is`) vs Equality (`==`) ⚠️

> 🧠 **Crucial Distinction:**
> - `==` checks **Equality of Values** (Do both objects contain the same data?).
> - `is` checks **Identity / Memory Location** (Do both variables point to the exact same object in memory `id(a) == id(b)`?).

```python
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print(list1 == list2)  # True  (Same values)
print(list1 is list2)  # False (Different memory allocations!)

# Small Integer Caching (-5 to 256 in CPython)
a = 100
b = 100
print(a is b)  # True (CPython reuses memory for small integers)
```

---

### 1.5 Membership Operators (`in`, `not in`)

```python
text = "Python Programming"
numbers = [10, 20, 30, 40]

print("Python" in text)    # True
print("Java" not in text)  # True
print(50 in numbers)       # False
```

---

## 📖 Topic 2: Type Conversion (Casting)

### 2.1 Implicit Type Conversion (Coercion)
Python automatically converts narrower types to wider types to prevent data loss.

```python
num_int = 10    # int
num_float = 2.5 # float

result = num_int + num_float
print(result)       # 12.5
print(type(result)) # <class 'float'>
```

### 2.2 Explicit Type Conversion (Type Casting)

```python
# 1. String to Int / Float
s = "100"
num = int(s)      # 100

# 2. Float to Int (Truncates decimal part, does not round!)
pi = 3.99
pi_int = int(pi)  # 3

# 3. Truthy and Falsy values using bool()
# Truthy: Non-zero numbers, non-empty strings/lists/dicts
# Falsy: 0, 0.0, "", [], {}, None, False
print(bool(0))        # False
print(bool(""))       # False
print(bool("Hello"))  # True
```

---

## ⚡ Master Cheat Sheet & Operator Precedence

### Operator Precedence (Highest to Lowest):
1. `()` — Parentheses
2. `**` — Exponentiation
3. `+x`, `-x` — Unary Plus / Minus
4. `*`, `/`, `//`, `%` — Multiplication, Division, Floor Div, Modulo
5. `+`, `-` — Addition, Subtraction
6. `==`, `!=`, `<`, `>`, `<=`, `>=`, `is`, `in` — Comparisons & Membership
7. `not` — Logical NOT
8. `and` — Logical AND
9. `or` — Logical OR

---

## ⚠️ Common Pitfalls & Best Practices

1. **Confusing `=` and `==`:**
   - ❌ `=` is assignment (`x = 5`).
   - ✅ `==` is comparison (`if x == 5:`).

2. **Truncation in `int()` vs `round()`:**
   - `int(4.9)` returns `4` (truncates decimal).
   - `round(4.9)` returns `5`.

3. **Chained Comparisons:**
   - Python supports intuitive chained comparisons: `18 <= age < 65` (equivalent to `age >= 18 and age < 65`).

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the output of `print(3 * 1 ** 3)`?
**Answer:** `3`. Exponentiation `**` has higher precedence than multiplication `*`. $1^3 = 1$, then $3 \times 1 = 3$.

### Q2: What is the difference between `list1 == list2` and `list1 is list2`?
**Answer:** `==` compares value contents. `is` checks whether both variables reference the exact same memory address (`id()`).

### Q3: What are "Falsy" values in Python?
**Answer:** `False`, `None`, numeric zeroes (`0`, `0.0`), and empty collections (`""`, `[]`, `()`, `{}`, `set()`).

---

## 📝 Recap Checklist
- [x] Mastered all 7 arithmetic operators including `/` vs `//` and `%`.
- [x] Differentiated value equality `==` from memory identity `is`.
- [x] Mastered `and`, `or`, `not` short-circuit logic.
- [x] Understood implicit type promotion and explicit type casting.

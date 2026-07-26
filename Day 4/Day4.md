# 🐍 Day 4/200 – Masterclass Notes: Loops & Control Flow (`for`, `while`, `break`, `continue`, `pass`)

🎯 **Goal:** Master iteration using `for` and `while` loops, range generation, control statements (`break`, `continue`, `pass`), infinite loop prevention, and loop `else` clauses.

---

## 📌 Executive Summary & Key Takeaways

- **Why Loops?** Avoid redundant code (DRY - Don't Repeat Yourself) by repeating execution blocks dynamically.
- **`for` Loop:** Definite iteration (iterates over sequences like `range()`, lists, strings).
- **`while` Loop:** Indefinite iteration (repeats as long as a condition evaluates to `True`).
- **`break`:** Immediately terminates the loop.
- **`continue`:** Skips the rest of the current iteration and jumps to the next cycle.
- **`pass`:** A null statement placeholder (does nothing; prevents syntax errors).

---

## 📖 Topic 1: `for` Loops & `range()`

### 1.1 `for` Loop Syntax
Used to iterate over any iterable object (sequence).

```python
# Iterating over range
for i in range(5):
    print(f"Iteration {i}")  # Prints 0, 1, 2, 3, 4
```

### 1.2 Deep Dive into `range(start, stop, step)`

| Call Signature | Start | Stop (Exclusive) | Step | Generated Numbers |
|---|:---:|:---:|:---:|---|
| `range(5)` | `0` | `5` | `1` | `0, 1, 2, 3, 4` |
| `range(1, 11)` | `1` | `11` | `1` | `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| `range(2, 21, 2)` | `2` | `21` | `2` | `2, 4, 6, 8, 10, 12, 14, 16, 18, 20` (Evens) |
| `range(10, 0, -1)` | `10` | `0` | `-1` | `10, 9, 8, 7, 6, 5, 4, 3, 2, 1` (Reverse) |

---

## 📖 Topic 2: `while` Loops & Infinite Loops

### 2.1 `while` Loop Syntax
Repeats a block of code as long as the test condition remains `True`.

```python
count = 1

while count <= 5:
    print(f"Count: {count}")
    count += 1  # ⚠️ Increment step is mandatory to avoid infinite loops!
```

### 2.2 Infinite Loops & How to Avoid Them
An infinite loop occurs when the loop condition never becomes `False`.

```python
# ❌ DANGER: Infinite loop (count never changes!)
# count = 1
# while count <= 5:
#     print(count)

# Useful Controlled Infinite Loop (CLI Menu pattern)
while True:
    user_input = input("Enter 'exit' to quit: ")
    if user_input.lower() == "exit":
        break  # Exit loop cleanly
```

---

## 📖 Topic 3: Loop Control Statements (`break`, `continue`, `pass`)

### 3.1 `break` Statement
Terminates the loop completely and moves execution to the statement immediately following the loop.

```python
for i in range(10):
    if i == 5:
        break  # Stops loop when i reaches 5
    print(i)   # Prints 0, 1, 2, 3, 4
```

### 3.2 `continue` Statement
Skips the remainder of the current loop iteration and moves directly to the next iteration cycle.

```python
for i in range(6):
    if i == 3:
        continue  # Skips printing 3
    print(i)      # Prints 0, 1, 2, 4, 5
```

### 3.3 `pass` Statement
A placeholder statement used when syntax requires a statement but no action needs to be taken.

```python
for i in range(5):
    if i == 2:
        pass  # TODO: Handle special case later
    print(i)
```

---

## 📖 Topic 4: `else` Clause with Loops (Advanced)
Python loops have a unique feature: an `else` block!
- The loop `else` block executes **ONLY IF** the loop finishes naturally without encountering a `break` statement.

```python
# Searching for a prime number / item
numbers = [2, 4, 6, 8]

for num in numbers:
    if num % 2 != 0:
        print("Found odd number!")
        break
else:
    # Executed because loop finished without hitting 'break'
    print("All numbers in list are even.")
```

---

## ⚡ Master Cheat Sheet & Control Flow Matrix

| Keyword | Action | Skips Remaining Loop Code? | Terminates Loop? |
|---|---|:---:|:---:|
| `break` | Aborts loop completely | ✅ Yes | ✅ Yes |
| `continue` | Jumps to next iteration | ✅ Yes | ❌ No |
| `pass` | Null operation / Placeholder | ❌ No | ❌ No |

---

## ⚠️ Common Pitfalls & Best Practices

1. **Off-by-One Errors in `range()`:**
   - Remember that `range(1, 10)` generates numbers up to `9`, NOT `10`!
   - To include `10`, use `range(1, 11)`.

2. **Forgetting State Counter Update in `while`:**
   - Forgetting `i += 1` inside a `while` loop creates a memory-consuming infinite loop.

3. **Modifying an Iterable While Looping Over It:**
   - ❌ `for item in my_list: my_list.remove(item)` (Causes skipped elements).
   - ✅ `for item in my_list.copy(): my_list.remove(item)`

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the output of `for i in range(1, 5, 2): print(i)`?
**Answer:** `1` and `3`. Starts at `1`, steps by `2`, stops before `5`.

### Q2: When does the `else` block attached to a `for` or `while` loop run?
**Answer:** The `else` block executes when the loop completes all iterations naturally. It does **NOT** run if the loop was exited prematurely using a `break` statement.

### Q3: What is the difference between `break` and `continue`?
**Answer:** `break` exits the entire loop immediately. `continue` skips only the rest of the current iteration and advances to the next iteration.

---

## 📝 Recap Checklist
- [x] Mastered `for` loops with `range(start, stop, step)`.
- [x] Mastered `while` loops and infinite loop control.
- [x] Applied `break`, `continue`, and `pass` in control flow.
- [x] Understood loop `else` clauses for search algorithm patterns.

# 🐍 Day 3/200 – Masterclass Notes: Conditional Statements (Control Flow)

🎯 **Goal:** Master Python decision-making using `if`, `if-else`, `if-elif-else`, nested conditionals, ternary expressions, logical conditions, and edge cases.

---

## 📌 Executive Summary & Key Takeaways

- **Control Flow:** Conditionals alter execution path based on truth values (`True`/`False`).
- **Indentation:** Python uses 4-space indentation blocks to demarcate body statements instead of `{}`.
- **Ternary Operator:** One-line conditional expression (`val_if_true if condition else val_if_false`).
- **Short-Circuit Evaluation:** Logical `and`/`or` stop evaluating as soon as the outcome is determined.

---

## 📖 Topic 1: Conditional Structures

### 1.1 Simple `if` Statement
Executes a block of code only if the condition evaluates to `True`.

```python
age = 20

if age >= 18:
    print("Eligible to vote")  # Executed because age >= 18 is True
```

---

### 1.2 `if-else` Statement
Provides an alternative branch when the `if` condition evaluates to `False`.

```python
age = 16

if age >= 18:
    print("Eligible to vote")
else:
    print("Not Eligible to vote")
```

---

### 1.3 `if-elif-else` Ladder
Used for checking multiple mutually exclusive conditions sequentially. Execution stops at the **first** `True` branch!

```python
marks = 85

if marks >= 90:
    print("Grade: A+")
elif marks >= 80:
    print("Grade: A")    # Executed! Next branches are skipped.
elif marks >= 70:
    print("Grade: B")
else:
    print("Need Improvement")
```

---

### 1.4 Nested `if` Statements
A conditional block inside another conditional block. Used when a secondary condition depends on a primary condition passing.

```python
age = 20
citizen = True

if age >= 18:
    if citizen:
        print("Access Granted: Can Vote!")
    else:
        print("Access Denied: Must be a citizen to vote.")
else:
    print("Access Denied: Underage.")
```

---

## 📖 Topic 2: Advanced Conditional Features

### 2.1 Ternary Operator (Conditional Expression)
A compact syntax for simple conditional assignments.

**Syntax:** `value_if_true if condition else value_if_false`

```python
age = 20
status = "Adult" if age >= 18 else "Minor"
print(status)  # Output: Adult
```

### 2.2 Combining Conditions with Logical Operators

```python
age = 25
salary = 45000
credit_score = 750

# Logical AND (All conditions must be True)
if age >= 21 and salary >= 30000 and credit_score >= 700:
    print("Loan Approved!")
else:
    print("Loan Rejected.")

# Logical OR (At least one condition must be True)
is_admin = False
is_owner = True

if is_admin or is_owner:
    print("Access Granted to Dashboard.")
```

---

## ⚡ Master Cheat Sheet & Decision Flowchart

```text
               ┌────────────────┐
               │ Check Condition│
               └───────┬────────┘
                       │
             True ┌────┴────┐ False
                  ▼         ▼
             [if branch] [else branch]
```

```python
# Quick Ref: Ternary Syntax
result = "Pass" if score >= 40 else "Fail"
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Incorrect Ordering in `elif` Chains:**
   - ❌ Putting broader conditions first:
     ```python
     if marks >= 50:  # Triggers for 95! Prevents checking >= 90
         print("Pass")
     elif marks >= 90:
         print("Grade A")
     ```
   - ✅ Order conditions from most restrictive to least restrictive (`>= 90`, then `>= 80`, then `>= 50`).

2. **Redundant Comparisons:**
   - ❌ `if is_logged_in == True:`
   - ✅ `if is_logged_in:`

3. **Overusing Deeply Nested Conditionals:**
   - ❌ Nesting 4-5 levels deep makes code unreadable (Pyramid of Doom).
   - ✅ Use guard clauses or `elif` ladders to keep code flat.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the output of `"Even" if 10 % 2 == 0 else "Odd"`?
**Answer:** `"Even"`. `10 % 2` is `0`, so the condition `0 == 0` evaluates to `True`.

### Q2: How does Python handle `elif` chain execution?
**Answer:** Python evaluates conditions top-to-bottom. As soon as one condition evaluates to `True`, its corresponding block runs and Python skips evaluating all remaining `elif` and `else` blocks in that ladder.

### Q3: What happens if none of the `if` or `elif` conditions evaluate to `True` and there is no `else` block?
**Answer:** Nothing happens; execution simply continues to the next statement outside the conditional block.

---

## 📝 Recap Checklist
- [x] Mastered `if`, `if-else`, and `if-elif-else` ladders.
- [x] Know how to structure nested conditionals cleanly.
- [x] Mastered ternary operator syntax `a if cond else b`.
- [x] Avoided common pitfall of incorrect ordering in `elif` chains.

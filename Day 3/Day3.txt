# 🐍 Day 3/200 – Conditional Statements (Control Flow)

🎯 **Goal:** Learn how to make decisions in your code using conditional statements, logical operators, and comparison operators.

---

## 1. The `if` Statement
The `if` statement is the simplest decision-making statement. It is used to decide whether a certain block of code should be executed based on a condition. 
- If the condition evaluates to `True`, the code block inside the `if` statement executes.
- If it evaluates to `False`, the code block is skipped.

*Note: Python uses indentation (usually 4 spaces) to define code blocks instead of curly braces `{}`.*

### Example:
```python
age = 20

if age >= 18:
    print("Eligible to vote")  # Output: Eligible to vote
```

---

## 2. The `if-else` Statement
The `if-else` statement evaluates a condition and provides an alternative action.
- If the condition is `True`, the `if` block executes.
- If the condition is `False`, the `else` block executes.

### Example:
```python
age = 16

if age >= 18:
    print("Eligible")
else:
    print("Not Eligible")  # Output: Not Eligible
```

---

## 3. The `if-elif-else` Ladder
When you have multiple conditions to check, you use `elif` (short for *else if*).
- Python checks the conditions sequentially from top to bottom.
- Once it finds a condition that is `True`, it executes that block and skips the rest.
- If none of the conditions are `True`, the `else` block executes.

### Example:
```python
marks = 85

if marks >= 90:
    print("Grade A+")
elif marks >= 80:
    print("Grade A")       # Output: Grade A
elif marks >= 70:
    print("Grade B")
else:
    print("Need Improvement")
```

---

## 4. Nested `if` Statements
A nested `if` statement is an `if` statement inside another `if` statement. This is useful when a second condition depends on the first condition being true.

### Example:
```python
age = 20
citizen = True

if age >= 18:
    if citizen:
        print("Can Vote")  # Output: Can Vote
```

---

## 5. Logical Operators
Logical operators are used to combine multiple conditional statements.

- **`and`**: Returns `True` if **both** conditions are true.
- **`or`**: Returns `True` if **at least one** condition is true.
- **`not`**: Inverts the condition (returns `False` if the condition is true, and vice versa).

### Example:
```python
age = 22
salary = 40000

# Both conditions must be True for the loan to be approved
if age >= 18 and salary >= 30000:
    print("Loan Approved")  # Output: Loan Approved
```

---

## 6. Comparison Operators
Comparison operators are used to compare two values. They return a boolean value (`True` or `False`).

| Operator | Meaning | Example | Result (if x = 5, y = 3) |
| :---: | :--- | :---: | :---: |
| **`==`** | Equal to | `x == y` | `False` |
| **`!=`** | Not equal to | `x != y` | `True` |
| **`>`** | Greater than | `x > y` | `True` |
| **`<`** | Less than | `x < y` | `False` |
| **`>=`** | Greater than or equal to | `x >= y` | `True` |
| **`<=`** | Less than or equal to | `x <= y` | `False` |

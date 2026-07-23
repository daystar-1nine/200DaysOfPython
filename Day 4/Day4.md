# 🐍 Day 4/200 – Loops and Control Statements

🎯 **Goal:** Learn how to repeat tasks efficiently using loops and control the flow of execution with `break`, `continue`, and `pass`.

---

## 1. Why Loops?
In programming, we often need to repeat a block of code multiple times. Writing the same statement over and over is inefficient and hard to maintain.

**Without Loops:**
```python
print("Hello")
print("Hello")
print("Hello")
print("Hello")
print("Hello")
```

**With Loops:**
```python
for i in range(5):
    print("Hello")
```

---

## 2. The `for` Loop
A `for` loop is used to iterate over a sequence (like a list, tuple, string, or range of numbers). It is ideal when you know in advance how many times the loop should run.

### Example:
```python
for i in range(5):
    print(i)
```

**Output:**
```text
0
1
2
3
4
```

---

## 3. The `range()` Function
The `range()` function generates a sequence of numbers. It can take up to three arguments: `range(start, stop, step)`.
- **`range(stop)`**: Generates numbers from `0` up to (but not including) `stop`.
- **`range(start, stop)`**: Generates numbers from `start` up to (but not including) `stop`.
- **`range(start, stop, step)`**: Generates numbers from `start` up to (but not including) `stop`, incrementing by `step`.

### Practice Examples:

- **Generate 1 to 10:**
  ```python
  for i in range(1, 11):
      print(i, end=" ")
  # Output: 1 2 3 4 5 6 7 8 9 10
  ```

- **Generate 10 to 1 (Reverse):**
  ```python
  for i in range(10, 0, -1):
      print(i, end=" ")
  # Output: 10 9 8 7 6 5 4 3 2 1
  ```

- **Generate Even Numbers (up to 20):**
  ```python
  for i in range(2, 21, 2):
      print(i, end=" ")
  # Output: 2 4 6 8 10 12 14 16 18 20
  ```

- **Generate Odd Numbers (up to 20):**
  ```python
  for i in range(1, 20, 2):
      print(i, end=" ")
  # Output: 1 3 5 7 9 11 13 15 17 19
  ```

---

## 4. The `while` Loop
A `while` loop repeats a block of code as long as a specified condition remains `True`. It is ideal when you don't know the exact number of iterations beforehand.

### Example:
```python
count = 1

while count <= 5:
    print(count)
    count += 1
```

**Output:**
```text
1
2
3
4
5
```

---

## 5. Infinite Loops
An infinite loop is a loop that never terminates because its condition is always `True`.

### Example of an Infinite Loop:
```python
while True:
    print("Running...")
```

### Why it happens:
The condition `True` is static and never changes. If a loop does not have a mechanism to change its condition to `False` (or hit a `break` statement), it will run forever, consuming CPU resources.

### How to avoid it:
- Always ensure the loop condition is updated within the body of the loop (e.g., incrementing a counter like `count += 1`).
- Provide an exit condition with a `break` statement.

---

## 6. The `break` Statement
The `break` statement is used to exit/terminate the loop prematurely, skipping any remaining iterations, even if the loop condition is still true.

### Example:
```python
for i in range(10):
    if i == 5:
        break  # Stops the loop completely when i reaches 5
    print(i)
```

**Output:**
```text
0
1
2
3
4
```

---

## 7. The `continue` Statement
The `continue` statement skips the current iteration and jumps directly to the next iteration of the loop.

### Example:
```python
for i in range(6):
    if i == 3:
        continue  # Skips printing 3 and moves to i = 4
    print(i)
```

**Output:**
```text
0
1
2
4
5
```

---

## 8. The `pass` Statement
The `pass` statement is a null operation. It does nothing. It is used as a placeholder when a statement is syntactically required, but you don't want to execute any code yet.

### Example:
```python
for i in range(5):
    if i == 2:
        pass  # Placeholder: do nothing when i is 2
    print(i)
```

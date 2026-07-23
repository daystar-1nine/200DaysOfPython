# 🐍 Day 2/200 – Variables, Operators & Type Conversion

🎯 **Goal:** Understand how Python stores data, performs calculations, and converts between data types.

---

## 1. Variables
Variables act as named containers for storing data values in memory. 

### Creating Variables:
In Python, variables are created the moment you assign a value to them using the assignment operator (`=`):
```python
name = "Suraj"
age = 20
cgpa = 8.85
```

### Variable Naming Rules:
- A variable name must start with a letter or the underscore character (`_`).
- A variable name cannot start with a number.
- A variable name can only contain alpha-numeric characters and underscores (`A-z`, `0-9`, and `_`).
- Variable names are case-sensitive (`age`, `Age`, and `AGE` are three different variables).
- Cannot use Python reserved keywords (such as `if`, `for`, `class`, `import`, etc.).

### Reassigning Values:
Python variables are dynamic. You can change their value (and type) at any time:
```python
x = 10       # x is an integer
x = "Hello"  # x is now a string
```

---

## 2. Data Types & `type()` Function
Python automatically identifies data types based on the value assigned.

Common data types:
- **`int`**: Integer (e.g., `20`, `-5`)
- **`float`**: Floating-point number (e.g., `8.85`, `3.14`)
- **`str`**: String / Text (e.g., `"Suraj"`, `'Python'`)
- **`bool`**: Boolean (`True` or `False`)

### Checking Data Types using `type()`:
You can check the data type of any variable using the built-in `type()` function:
```python
name = "Suraj"
age = 20

print(type(name))  # Output: <class 'str'>
print(type(age))   # Output: <class 'int'>
```

---

## 3. Operators
Operators are special symbols used to perform operations on variables and values.

### A. Arithmetic Operators
Used for mathematical calculations:
- `+` (Addition): `10 + 5` -> `15`
- `-` (Subtraction): `10 - 5` -> `5`
- `*` (Multiplication): `10 * 5` -> `50`
- `/` (Division - returns float): `10 / 4` -> `2.5`
- `%` (Modulus - remainder): `10 % 3` -> `1`
- `**` (Exponentiation - power): `2 ** 3` -> `8`
- `//` (Floor Division - rounds down to whole number): `10 // 4` -> `2`

### B. Comparison Operators
Used to compare two values. Always returns a boolean (`True` or `False`):
- `==` (Equal to): `5 == 5` -> `True`
- `!=` (Not equal to): `5 != 3` -> `True`
- `>` (Greater than): `10 > 5` -> `True`
- `<` (Less than): `5 < 2` -> `False`
- `>=` (Greater than or equal to): `5 >= 5` -> `True`
- `<=` (Less than or equal to): `4 <= 5` -> `True`

### C. Assignment Operators
Used to assign values to variables:
- `=` : `x = 5`
- `+=`: `x += 3` (equivalent to `x = x + 3`)
- `-=`: `x -= 2` (equivalent to `x = x - 2`)
- `*=`: `x *= 4` (equivalent to `x = x * 4`)

### D. Logical Operators
Used to combine conditional statements:
- `and`: Returns `True` if both statements are true (`x > 3 and x < 10`)
- `or`: Returns `True` if at least one statement is true (`x > 5 or x < 4`)
- `not`: Reverse the result, returns `False` if the result is true (`not(x > 3 and x < 10)`)

---

## 4. Type Conversion (Typecasting)
Converting a variable from one data type to another.

### Built-in Type Conversion Functions:
- `int()`: Converts value to an integer.
- `float()`: Converts value to a floating-point number.
- `str()`: Converts value to a string.
- `bool()`: Converts value to a boolean.

### Practical Example:
The `input()` function always returns input as a string (`str`). To perform calculations, you must convert it to `int` or `float`:
```python
age = input("Enter age: ")  # User enters "20" -> age is '20' (str)
age = int(age)              # Converted to 20 (int)
print("Next year you will be:", age + 1)
```

---

## 5. Basic Math Functions
Python provides several built-in mathematical functions:

- **`abs(x)`**: Returns the absolute (positive) value of a number.
  ```python
  print(abs(-7.5))  # Output: 7.5
  ```

- **`round(number, ndigits)`**: Rounds a number to a specified number of decimals (default is 0).
  ```python
  print(round(3.14159, 2))  # Output: 3.14
  print(round(4.7))         # Output: 5
  ```

- **`pow(base, exp)`**: Calculates base raised to the power of exp (same as `base ** exp`).
  ```python
  print(pow(2, 3))  # Output: 8
  ```

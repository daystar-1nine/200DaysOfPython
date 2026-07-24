# 🐍 Day 5/200 – Functions in Python

🎯 **Goal:** Learn how to write reusable code using functions and understand parameters, arguments, return values, and variable scope.

---

## 1. What is a Function?
A function is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing.

### Example:
```python
# Defining the function
def greet():
    print("Hello, Welcome to Python!")

# Calling the function
greet()  # Output: Hello, Welcome to Python!
```

---

## 2. Function Syntax
In Python, functions are defined using the `def` keyword followed by the function name, parentheses `()`, and a colon `:`.

```python
def function_name(parameters):
    """Docstring: Optional description of what the function does"""
    # code block
    # return statement (optional)
```

---

## 3. Functions with Parameters
**Parameters** are the placeholders defined in the function signature. **Arguments** are the actual values passed to the function when it is called.

### Example:
```python
def greet(name):  # 'name' is a parameter
    print(f"Hello, {name}!")

greet("Suraj")  # "Suraj" is an argument
# Output: Hello, Suraj!
```

---

## 4. Functions with Multiple Parameters
You can pass multiple parameters by separating them with commas.

### Example:
```python
def add(a, b):
    print(f"Sum: {a + b}")

add(10, 20)  # Output: Sum: 30
```

---

## 5. The `return` Statement
The `return` statement is used to exit a function and send a value back to the caller. If no `return` statement is defined, the function returns `None` by default.

### Example:
```python
def square(number):
    return number * number

result = square(5)
print(result)  # Output: 25
```

---

## 6. Default Parameters
You can assign default values to parameters. If no argument is provided for that parameter during the function call, the default value is used.

### Example:
```python
def greet(name="Guest"):
    print(f"Hello, {name}!")

greet()        # Output: Hello, Guest! (uses default)
greet("Suraj") # Output: Hello, Suraj! (overrides default)
```

---

## 7. Variable Scope
Variable scope refers to where in the program a variable is accessible.

### A. Local Variables
Variables defined inside a function. They can only be accessed within that function.

### B. Global Variables
Variables defined outside any function. They can be accessed anywhere in the file.

### Example:
```python
x = 100  # Global variable

def demo():
    y = 50  # Local variable
    print(f"Inside function (local y): {y}")
    print(f"Inside function (global x): {x}")

demo()
print(f"Outside function (global x): {x}")

# print(y)  # Error! NameError: name 'y' is not defined (y is local to demo)
```

---

## 8. Why Use Functions?
- **Code Reusability:** Write once, run many times. Avoids duplicate code.
- **Improved Readability:** Breaks long, complex programs into small, logical sections.
- **Easier Debugging:** If something goes wrong, you only need to fix the logic inside the specific function.
- **Better Organization:** Helps keep code clean and structured.

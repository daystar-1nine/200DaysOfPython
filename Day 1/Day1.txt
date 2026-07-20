# Day 1: Python Basics & Setup

Welcome to Day 1 of your Python journey! Below is the detailed explanation of all the topics for today.

---

## 1. What is Python?
Python is a high-level, interpreted, general-purpose programming language. It was created by Guido van Rossum and first released in 1991. 
- **High-level:** Its syntax is close to human language, making it easy to read and write.
- **Interpreted:** Python code is executed line-by-line by an interpreter, meaning you don't need to compile it before running.
- **General-purpose:** You can use it for web development, data science, machine learning, automation, scripting, and more.

---

## 2. Why Python is Popular
- **Easy to Learn & Read:** Clean syntax that resembles plain English.
- **Large Community & Libraries:** Thousands of pre-built packages (like NumPy, Pandas, Django) to solve complex problems easily.
- **Versatility:** Used in various fields, from small automation scripts to powering giant platforms like Instagram, Google, and Netflix.
- **Cross-platform:** Runs seamlessly on Windows, macOS, and Linux.

---

## 3. Installing Python, VS Code, and PyCharm
### Installing Python:
1. Go to the official website: https://www.python.org/
2. Download the latest version for your OS.
3. **Important (Windows):** Check the box that says **"Add Python to PATH"** during installation.

### Installing IDEs (Integrated Development Environments):
- **VS Code (Visual Studio Code):** A lightweight, highly customizable code editor. Download it from https://code.visualstudio.com/ and install the "Python" extension by Microsoft.
- **PyCharm:** A full-featured IDE built specifically for Python by JetBrains. Download the free "Community Edition" from https://www.jetbrains.com/pycharm/.

---

## 4. Running Python from the Terminal
Open your terminal (Command Prompt, PowerShell, or macOS/Linux Terminal) and check if Python is installed:
```bash
python --version
```

### Interactive Shell (REPL):
Type `python` (or `python3` on Mac/Linux) and press Enter to start the interactive shell:
```python
>>> print("Hello, World!")
Hello, World!
```
To exit the shell, type `exit()` and press Enter.

---

## 5. Creating Your First Python File (hello.py)
1. Create a new file named `hello.py`.
2. Write the following code:
   ```python
   print("Hello, World!")
   ```
3. Run the file from your terminal by navigating to its folder and typing:
   ```bash
   python hello.py
   ```

---

## 6. Comments (`#`)
Comments are notes in the code that are ignored by the Python interpreter. They are used to explain what the code does.
```python
# This is a single-line comment
print("Hello")  # This prints Hello to the screen
```

---

## 7. The `print()` Function
Used to output data to the screen (standard output).
```python
print("Hello, World!")
print(123)
print("Result is:", 10 + 5)
```

---

## 8. The `input()` Function
Used to take input from the user. It always returns the user input as a string (`str`).
```python
name = input("Enter your name: ")
print("Hello, " + name)
```

---

## 9. Variables
Variables are containers for storing data values. You don't need to declare their type beforehand.
```python
age = 25          # Stores an integer
name = "Alice"    # Stores a string
pi = 3.14         # Stores a float
is_active = True  # Stores a boolean
```

---

## 10. Basic Data Types
- **`int` (Integer):** Whole numbers without a decimal point.
  ```python
  x = 10
  y = -5
  ```
- **`float` (Floating point number):** Numbers containing decimals.
  ```python
  x = 10.5
  y = -0.01
  ```
- **`str` (String):** Text wrapped in single (`'`) or double (`"`) quotes.
  ```python
  greeting = "Hello"
  char = 'A'
  ```
- **`bool` (Boolean):** Represents logical values: `True` or `False`.
  ```python
  is_logged_in = True
  has_passed = False
  ```

---

## 11. Basic Naming Conventions
When naming variables in Python, follow these rules:
- **Snake Case:** Use lowercase words separated by underscores (e.g., `user_age`, `total_price`).
- **Allowed Characters:** Must start with a letter or underscore (`_`). Can contain letters, numbers, and underscores (e.g., `value_1`).
- **Case-sensitive:** `age`, `Age`, and `AGE` are three different variables.
- **Reserved Keywords:** Cannot use Python keywords (like `print`, `input`, `if`, `while`, `class`) as variable names.

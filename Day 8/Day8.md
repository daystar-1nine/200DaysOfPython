# 🐍 Day 8/200 – Masterclass Notes: Strings & Text Processing in Python

🎯 **Goal:** Master Python strings, character indexing, slicing, string immutability, essential string methods, formatting techniques (f-Strings), escape sequences, input validation, and text analysis.

---

## 📌 Executive Summary & Key Takeaways

- **String (`str`):** An immutable, ordered sequence of Unicode characters enclosed in single (`'`), double (`"`), or triple (`"""`) quotes.
- **Immutability:** Strings cannot be modified in-place; any operation that modifies a string creates and returns a **new string object**.
- **String Methods:** Essential for text manipulation (`upper()`, `lower()`, `strip()`, `replace()`, `split()`, `join()`, `find()`, `count()`, `startswith()`, `endswith()`).
- **Input Validation:** Validation methods like `.isalpha()`, `.isdigit()`, `.isalnum()`, `.isspace()`.
- **f-Strings:** The fastest, cleanest, and modern standard for string interpolation in Python 3.6+.

---

## 📖 Topic 1: What is a String? (Representation & Immutability)

### 1.1 String Declarations
Strings can be defined using single, double, or triple quotes (multi-line strings/docstrings).

```python
single_quote = 'Hello'
double_quote = "Python"
multi_line = """This is a
multi-line string."""
```

### 1.2 String Immutability Deep Dive
In Python, strings are **immutable**. Once created, their individual characters cannot be reassigned.

```python
text = "Python"

# ❌ INCORRECT: Trying to mutate a character
# text[0] = "J"  # Raises TypeError: 'str' object does not support item assignment

# ✅ CORRECT: Create a new string by concatenation or replacement
text = "J" + text[1:]
print(text)  # "Jython"
```

---

## 📖 Topic 2: Character Indexing & Advanced Slicing

### 2.1 Indexing
Strings use 0-based positive indexing from left to right, and negative indexing from right to left (`-1` for the last character).

```python
text = "Programming"

print(text[0])   # 'P' (First character)
print(text[-1])  # 'g' (Last character)
```

### 2.2 Slicing Syntax (`text[start:stop:step]`)

```python
word = "Programming"

print(word[0:6])   # 'Progra' (Index 0 to 5)
print(word[3:])    # 'gramming' (Index 3 to end)
print(word[:5])    # 'Progr' (Start to index 4)
print(word[::2])   # 'Pgamn' (Every 2nd character)
print(word[::-1])  # 'gnimmargorP' (Reversed string)
```

---

## 📖 Topic 3: String Methods Deep Dive

### 3.1 Case Conversion Methods
| Method | Description | Example |
|---|---|---|
| `upper()` | Converts to uppercase | `"hello".upper() -> "HELLO"` |
| `lower()` | Converts to lowercase | `"HELLO".lower() -> "hello"` |
| `title()` | Capitalizes first letter of every word | `"hello world".title() -> "Hello World"` |
| `capitalize()` | Capitalizes only the first letter of string | `"hello world".capitalize() -> "Hello world"` |
| `swapcase()` | Swaps uppercase to lowercase and vice versa | `"PyThOn".swapcase() -> "pYtHoN"` |

---

### 3.2 Whitespace & Trimming Methods
| Method | Description | Example |
|---|---|---|
| `strip()` | Removes leading & trailing whitespace | `"  python  ".strip() -> "python"` |
| `lstrip()` | Removes leading (left) whitespace | `"  python".lstrip() -> "python"` |
| `rstrip()` | Removes trailing (right) whitespace | `"python  ".rstrip() -> "python"` |

---

### 3.3 Searching & Counting Methods
| Method | Description | Return Value |
|---|---|---|
| `find(sub)` | Finds first occurrence of substring | Index if found, **`-1` if not found** |
| `rfind(sub)` | Finds last occurrence of substring | Index if found, `-1` if not found |
| `index(sub)` | Finds first occurrence of substring | Index if found, **Raises `ValueError` if missing!** |
| `count(sub)` | Counts occurrences of substring | Integer count |
| `startswith(sub)` | Checks if string starts with substring | `True` / `False` |
| `endswith(sub)` | Checks if string ends with substring | `True` / `False` |

```python
text = "Python Programming"

print(text.find("gram"))      # 10
print(text.find("Java"))      # -1 (Safe!)
print(text.count("o"))        # 2
print(text.startswith("Py"))  # True
print(text.endswith(".py"))   # False
```

---

### 3.4 Splitting & Joining Methods

- **`split(sep)`:** Splits string into a list of substrings using delimiter `sep` (defaults to whitespace).
- **`join(iterable)`:** Joins elements of an iterable of strings into a single string using a separator.

```python
# Splitting string into list
sentence = "Python,Java,C++,JavaScript"
languages = sentence.split(",")
print(languages)  # ['Python', 'Java', 'C++', 'JavaScript']

# Joining list into string
joined_text = " | ".join(languages)
print(joined_text)  # "Python | Java | C++ | JavaScript"
```

---

### 3.5 Character Validation Methods (Boolean Checkers)

| Method | Returns `True` If... |
|---|---|
| `isalpha()` | All characters are alphabetic letters ($a-z, A-Z$) |
| `isdigit()` | All characters are digits ($0-9$) |
| `isalnum()` | All characters are alphanumeric (letters or digits) |
| `isspace()` | All characters are whitespace (` `, `\t`, `\n`) |
| `islower()` | All cased characters are lowercase |
| `isupper()` | All cased characters are uppercase |

```python
"Python".isalpha()  # True
"12345".isdigit()   # True
"User123".isalnum() # True
"   ".isspace()     # True
```

---

## 📖 Topic 4: String Formatting (f-Strings)

Modern Python 3.6+ utilizes **f-Strings (Formatted String Literals)** for readable and fast string interpolation.

```python
name = "Suraj"
age = 20
score = 98.4567

# Basic f-String
print(f"Name: {name}, Age: {age}")

# Expressions inside f-Strings
print(f"Next year: {age + 1}")
print(f"Uppercase Name: {name.upper()}")

# Number Formatting & Precision (e.g. 2 decimal places)
print(f"Score: {score:.2f}")  # Output: 98.46

# Alignment & Padding
print(f"{name:>10}")  # Right-align within 10 spaces
print(f"{name:<10}")  # Left-align within 10 spaces
print(f"{name:^10}")  # Center-align within 10 spaces
```

---

## 📖 Topic 5: Escape Characters & Raw Strings

| Escape Sequence | Description | Output Example |
|---|---|---|
| `\n` | Newline | Breaks line |
| `\t` | Tab space | Adds horizontal tab |
| `\"` | Double quote | Includes literal `"` |
| `\'` | Single quote | Includes literal `'` |
| `\\` | Backslash | Includes literal `\` |

### Raw Strings (`r"..."`)
Prefixing a string with `r` disables escape sequence interpretation (useful for regex & file paths):

```python
# Standard String (Interprets \n as newline)
print("C:\new_folder\test.txt")

# Raw String (Treats backslashes literally)
print(r"C:\new_folder\test.txt")  # C:\new_folder\test.txt
```

---

## ⚡ Master Cheat Sheet & Quick Summary

```python
# Common String Operations Cheat Sheet
s = "  Python Programming  "

s_clean = s.strip()            # "Python Programming"
s_upper = s_clean.upper()       # "PYTHON PROGRAMMING"
s_rep = s_clean.replace("P", "J") # "Jython Jrogramming"
words = s_clean.split(" ")      # ['Python', 'Programming']
reversed_s = s_clean[::-1]      # "gnimmargorP nohtyP"
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **`find()` vs `index()`:**
   - Use `.find()` when searching for optional substrings because it returns `-1` if missing.
   - Using `.index()` raises `ValueError` if the substring is missing.

2. **Joining Non-String Iterables:**
   - ❌ `", ".join([1, 2, 3])` (Raises `TypeError: expected str instance, int found`).
   - ✅ `", ".join([str(x) for x in [1, 2, 3]])`

3. **String Concatenation in Loops:**
   - ❌ Avoid `s += char` inside large loops ($O(n^2)$ time complexity due to creating $n$ new strings).
   - ✅ Append to a list `lst.append(char)` and use `"".join(lst)` ($O(n)$ time complexity).

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: Why are strings immutable in Python?
**Answer:** String immutability offers several advantages:
1. **Security & Hashability:** Strings can be safely used as dictionary keys and set elements because their hash values never change.
2. **Memory Optimization (String Interning):** Python reuses existing immutable string objects in memory to save RAM.
3. **Thread Safety:** Immutable objects are inherently thread-safe in concurrent applications.

### Q2: How do you check if a string is a palindrome in Python?
**Answer:** Compare the cleaned string with its reverse slice `text == text[::-1]`.

### Q3: What is the difference between `s.strip()` and `s.replace(" ", "")`?
**Answer:** `s.strip()` only removes spaces from the **beginning and end** of the string. `s.replace(" ", "")` removes **all spaces**, including spaces between words.

---

## 📝 Recap Checklist
- [x] Understand string immutability and character indexing.
- [x] Master slicing syntax `[start:stop:step]` and reverse slicing `[::-1]`.
- [x] Master essential methods (`upper()`, `strip()`, `split()`, `join()`, `replace()`, `find()`).
- [x] Leverage f-strings for dynamic string formatting and float rounding.
- [x] Validate string inputs using `.isalpha()`, `.isdigit()`, `.isalnum()`.

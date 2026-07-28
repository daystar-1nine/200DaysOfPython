# 🐍 Day 9/200 – Masterclass Notes: File Handling in Python

🎯 **Goal:** Master reading, writing, appending, updating, and managing files in Python using built-in file functions, context managers (`with`), error handling, and `os`/`pathlib` modules.

---

## 📌 Executive Summary & Key Takeaways

- **Persistence:** File handling enables Python programs to store data permanently on secondary storage (hard drives/SSDs) beyond program execution memory.
- **Context Manager (`with`):** The modern, industry-standard way to handle files. Automatically manages resource disposal and guarantees file closure even if exceptions occur.
- **File Modes:** `"r"` (Read), `"w"` (Write/Overwrite), `"a"` (Append), `"x"` (Exclusive creation), `"b"` (Binary mode for images/audio), `"+"` (Read/Write update).
- **Buffer Management:** Methods like `read()` load the entire file into memory; `readline()` or iterating over the file object reads line-by-line in a memory-efficient $O(1)$ stream.

---

## 📖 Topic 1: Opening & Closing Files

### 1.1 The `open()` Function Syntax
```python
file_object = open(file_path, mode="r", encoding="utf-8")
```

### 1.2 File Modes Matrix

| Mode | Name | Description | If File Exists | If File Missing |
|---|---|---|---|---|
| `"r"` | **Read** (Default) | Opens file for reading text | ✅ Opens file | ❌ Raises `FileNotFoundError` |
| `"w"` | **Write** | Opens file for writing text | ⚠️ **Overwrites / Truncates** | ✅ Creates new file |
| `"a"` | **Append** | Opens file for appending text at the end | ✅ Preserves content & appends | ✅ Creates new file |
| `"x"` | **Exclusive Create** | Creates a new file exclusively | ❌ Raises `FileExistsError` | ✅ Creates new file |
| `"r+"`| **Read & Write** | Opens file for both reading and writing | ✅ Opens file | ❌ Raises `FileNotFoundError` |
| `"b"` | **Binary** | Appended to modes for binary files (`"rb"`, `"wb"`) | Operates on raw bytes | Operates on raw bytes |

---

## 📖 Topic 2: Reading Files Efficiently

### 2.1 Reading Methods Overview

```python
# 1. read(size=-1): Reads entire file (or up to 'size' characters)
with open("sample.txt", "r") as f:
    content = f.read()

# 2. readline(): Reads a single line ending with newline '\n'
with open("sample.txt", "r") as f:
    line1 = f.readline()
    line2 = f.readline()

# 3. readlines(): Reads all lines into a list of strings
with open("sample.txt", "r") as f:
    lines_list = f.readlines()

# 4. Iterating directly over file object (BEST for large files / Low RAM)
with open("large_log.txt", "r") as f:
    for line in f:
        print(line.strip())
```

---

## 📖 Topic 3: Writing & Appending Files

### 3.1 Writing (`"w"`) vs Appending (`"a"`)

```python
# Mode 'w' overwrites existing data!
with open("output.txt", "w") as f:
    f.write("Header: User Log\n")
    f.write("Line 1: System Initialized\n")

# Mode 'a' appends new data to the end without deleting old data!
with open("output.txt", "a") as f:
    f.write("Line 2: New Event Occurred\n")
```

### 3.2 Writing Multiple Lines (`writelines()`)
```python
lines = ["First Line\n", "Second Line\n", "Third Line\n"]

with open("batch.txt", "w") as f:
    f.writelines(lines)
```

---

## 📖 Topic 4: Context Managers (`with` Statement - Best Practice)

Always use `with` statements when handling file I/O!

```python
# ❌ BAD PRACTICE (Manual open & close):
f = open("data.txt", "r")
data = f.read()
# If an error happens here, f.close() is NEVER called! (Memory/Resource Leak)
f.close()

# ✅ BEST PRACTICE (Context Manager):
with open("data.txt", "r") as f:
    data = f.read()
# File is automatically closed here even if an exception occurs!
```

---

## 📖 Topic 5: Checking File Existence & Management (`os` & `pathlib`)

```python
import os
from pathlib import Path

file_path = "notes.txt"

# Method 1: Using os module
if os.path.exists(file_path):
    print("File exists!")
    if os.path.isfile(file_path):
        print(f"File Size: {os.path.getsize(file_path)} bytes")

# Method 2: Using modern pathlib (Python 3.4+)
path = Path(file_path)
if path.is_file():
    print(f"Path exists: {path.name}")

# Removing / Deleting a File safely
if os.path.exists("temp.txt"):
    os.remove("temp.txt")
```

---

## ⚡ Master Cheat Sheet & File Operations Summary

```python
# Comprehensive File Handling Cheat Sheet
import os

filename = "demo.txt"

# Write
with open(filename, "w", encoding="utf-8") as f:
    f.write("Line 1\nLine 2\n")

# Append
with open(filename, "a", encoding="utf-8") as f:
    f.write("Line 3\n")

# Read Line by Line
if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            print(f"Line {line_num}: {line.strip()}")
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `encoding="utf-8"`:**
   - Always pass `encoding="utf-8"` to `open()` to prevent cross-platform text corruptions between Windows (cp1252) and Linux/macOS (utf-8).

2. **Loading Giant Files into RAM with `.read()`:**
   - ❌ `data = open("10GB_file.log").read()` (Crashes program with `MemoryError`).
   - ✅ Iterate line by line: `for line in open("10GB_file.log"): process(line)`.

3. **Confusing `"w"` and `"a"` modes:**
   - Mode `"w"` silently erases all existing content in the file! Use `"a"` if you want to keep existing data.

4. **Not stripping newline `\n` characters when reading lines:**
   - Use `line.strip()` when processing `f.readline()` or `for line in f:`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the main advantage of using `with open(...) as f:` over `f = open(...)`?
**Answer:** The `with` statement utilizes Python's context manager protocol (`__enter__` and `__exit__` methods). It guarantees that file handles are properly closed automatically as soon as execution leaves the `with` block, even if runtime exceptions or crashes occur.

### Q2: What is the difference between `.read()`, `.readline()`, and `.readlines()`?
**Answer:**
- `.read()`: Reads the entire file into a single string.
- `.readline()`: Reads a single line from the file stream.
- `.readlines()`: Reads all remaining lines into a Python list of strings.

### Q3: How do you safely check if a file exists before attempting to open it in read mode?
**Answer:** Use `os.path.isfile("path/to/file")` or `Path("path/to/file").is_file()`, or wrap the `open()` call inside a `try...except FileNotFoundError` block.

---

## 📝 Recap Checklist
- [x] Mastered file opening modes (`r`, `w`, `a`, `x`, `r+`, `b`).
- [x] Used context managers (`with open()`) for clean resource management.
- [x] Read files line-by-line to handle large files memory-efficiently.
- [x] Managed file existence checks using `os.path` and `pathlib`.

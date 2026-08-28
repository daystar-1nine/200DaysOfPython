# 🐍 Day 25/200 – Masterclass Notes: Python + SQLite Database

🎯 **Goal:** Master database storage in Python using embedded **SQLite (`sqlite3`)**, execute **CRUD operations** (Create, Read, Update, Delete), prevent **SQL Injection** using parameterized queries, manage database transactions (`commit`, `rollback`), use SQL aggregate functions (`COUNT`, `AVG`, `MAX`, `MIN`), and build CLI-driven database applications.

---

## 📌 Executive Summary & Key Takeaways

- **Why Relational Databases vs. JSON/CSV:** Flat files require rewriting the entire file for every change, lack atomic transaction safety, and do not scale efficiently for concurrent queries. Databases provide structured schemas, indexing, atomic ACID transactions, and optimized SQL queries.
- **Python's `sqlite3` Module:** Built into Python standard library; no external server installation required. Database stored as a single `.db` file on disk.
- **Connection & Cursor Lifecycle:**
  - `connection = sqlite3.connect("database.db")`: Establishes session to database file.
  - `cursor = connection.cursor()`: Executes SQL queries and traverses result sets.
- **The CRUD Matrix:**
  - **Create:** `INSERT INTO table (col1, col2) VALUES (?, ?)`
  - **Read:** `SELECT col1, col2 FROM table WHERE condition ORDER BY col1 DESC`
  - **Update:** `UPDATE table SET col1 = ? WHERE id = ?`
  - **Delete:** `DELETE FROM table WHERE id = ?`
- **Security Mandate - Parameterized Queries:** NEVER format raw SQL using f-strings (`f"SELECT * FROM users WHERE name = '{user}'"`). ALWAYS use `?` placeholders (`cursor.execute("SELECT * FROM users WHERE name = ?", (user,))`) to prevent **SQL Injection attacks**.

---

## 📖 Topic 1: Database Setup & Table Creation

```python
import sqlite3

# Context manager handles automatic commit and rollback
with sqlite3.connect("students.db") as conn:
    cursor = conn.cursor()
    # Create Table Schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER CHECK(age > 0),
            marks REAL DEFAULT 0.0
        )
    """)
```

---

## 📖 Topic 2: Parameterized CRUD Operations

### 2.1 Inserting & Batch Inserting

```python
# Single Insert
cursor.execute(
    "INSERT INTO students (name, age, marks) VALUES (?, ?, ?)",
    ("Suraj Sawant", 20, 88.5)
)

# Batch Insert (executemany)
student_records = [
    ("Rahul Sharma", 21, 91.0),
    ("Priya Patel", 19, 84.0),
    ("Amit Kumar", 22, 76.5)
]
cursor.executemany(
    "INSERT INTO students (name, age, marks) VALUES (?, ?, ?)",
    student_records
)
```

### 2.2 Queries, Aggregates & Filtering

```python
# Fetch All
cursor.execute("SELECT * FROM students ORDER BY marks DESC")
all_students = cursor.fetchall()

# Parameterized WHERE Filter
cursor.execute("SELECT name, marks FROM students WHERE marks > ?", (80.0,))
top_students = cursor.fetchall()

# SQL Aggregate Functions
cursor.execute("SELECT COUNT(*), AVG(marks), MAX(marks), MIN(marks) FROM students")
total_count, avg_marks, max_marks, min_marks = cursor.fetchone()
```

---

## 📖 Topic 3: Schema Design & Business Logic Constraints

```python
# Foreign Key Relationships & Status Tracking
cursor.execute("""
    CREATE TABLE IF NOT EXISTS borrowings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER NOT NULL,
        member_id INTEGER NOT NULL,
        borrowed_date TEXT NOT NULL,
        returned_date TEXT,
        FOREIGN KEY (book_id) REFERENCES books (id),
        FOREIGN KEY (member_id) REFERENCES members (id)
    )
""")
```

---

## ⚡ Master Cheat Sheet

```python
# SQLite & Python Master Cheat Sheet

import sqlite3

# 1. Connection & Table Init
with sqlite3.connect("app.db") as conn:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, qty INT)")

# 2. Parameterized Query Execution
def add_item(name: str, qty: int):
    with sqlite3.connect("app.db") as conn:
        conn.cursor().execute("INSERT INTO items (name, qty) VALUES (?, ?)", (name, qty))

# 3. Safe Row Fetching
def get_item_by_id(item_id: int):
    with sqlite3.connect("app.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, qty FROM items WHERE id = ?", (item_id,))
        return cur.fetchone()  # Returns tuple or None
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Constructing SQL Strings via F-Strings or Format Specifiers:**
   - ❌ `cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")` (Vulnerable to `' OR '1'='1` SQL Injection attacks).
   - ✅ `cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))`.

2. **Forgetting `connection.commit()`:**
   - Data modifications (`INSERT`, `UPDATE`, `DELETE`) remain in transient transaction memory and are discarded when connection closes unless committed.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the purpose of `AUTOINCREMENT` on a `PRIMARY KEY` in SQLite?
**Answer:** It ensures that when new rows are inserted without specifying an `id`, SQLite automatically assigns a monotonically increasing unique integer value that is never reused, even if previous rows were deleted.

### Q2: What is the difference between `fetchone()` and `fetchall()`?
**Answer:** `fetchone()` retrieves the next single row from the cursor result set as a tuple (or `None` if no rows remain). `fetchall()` retrieves all remaining matching rows from the result set as a list of tuples.

---

## 📝 Recap Checklist
- [x] Connected to SQLite database using `sqlite3.connect()`.
- [x] Defined SQL tables with primary keys, types, and default values.
- [x] Mastered CRUD operations (Create, Read, Update, Delete).
- [x] Applied parameterized queries (`?`) to prevent SQL Injection.
- [x] Executed SQL aggregate functions (`COUNT`, `AVG`, `MAX`, `MIN`).
- [x] Built Student Database CLI App, SQLite Expense Tracker, and Library Management System projects.

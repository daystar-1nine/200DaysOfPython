# 🐍 Day 26/200 – Masterclass Notes: Advanced SQL & Database Design

🎯 **Goal:** Master multi-table relational database design, Foreign Key relationships, `PRAGMA foreign_keys = ON;`, `INNER JOIN` and `LEFT JOIN`, `GROUP BY` with `HAVING`, database indexing (`CREATE INDEX`), database normalization fundamentals, and multi-statement atomic transactions (`commit` / `rollback`).

---

## 📌 Executive Summary & Key Takeaways

- **Database Normalization:** Organizing database schemas to eliminate redundant data storage across rows. Instead of embedding user details in every expense row, separate entity tables (`users`, `categories`, `expenses`) are connected via Foreign Keys.
- **Relational Schema Design:**
  - **Primary Key (`PRIMARY KEY`):** Unique identifier per row.
  - **Foreign Key (`FOREIGN KEY`):** References a Primary Key in a parent table.
  - **SQLite Constraint Enforcer:** SQLite disables Foreign Key constraints by default! Must execute `PRAGMA foreign_keys = ON;` on every new connection.
- **SQL JOIN Types:**
  - `INNER JOIN`: Returns matching records present in both connected tables.
  - `LEFT JOIN`: Returns all records from the left table, with NULLs for unmatched right table rows.
- **`GROUP BY` vs `HAVING`:**
  - `WHERE`: Filters individual records *before* aggregation occurs.
  - `GROUP BY`: Groups records by specified columns for aggregate functions (`SUM`, `COUNT`, `AVG`).
  - `HAVING`: Filters grouped summary records *after* aggregation (e.g. `HAVING SUM(amount) > 1000`).
- **Database Indexing:** `CREATE INDEX idx_name ON table(column)` creates B-Tree lookup structures accelerating `SELECT` queries on large datasets.

---

## 📖 Topic 1: Relational Schema & Foreign Keys

```python
import sqlite3

# Context manager handling connection
with sqlite3.connect("expense.db") as conn:
    # MANDATORY: Enable foreign key enforcement in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # 1. Parent Table: users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)

    # 2. Parent Table: categories
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    # 3. Child Table: expenses (referencing users and categories)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    """)
```

---

## 📖 Topic 2: Relational Queries & Aggregations

### 2.1 Multi-Table `INNER JOIN` with `GROUP BY` & `HAVING`

```python
# Select spending breakdown per user for totals exceeding Rs.5000
query = """
SELECT 
    users.name,
    categories.name AS category_name,
    SUM(expenses.amount) AS total_spent
FROM expenses
INNER JOIN users ON expenses.user_id = users.id
INNER JOIN categories ON expenses.category_id = categories.id
GROUP BY users.id, categories.id
HAVING SUM(expenses.amount) > 5000
ORDER BY total_spent DESC
"""

cursor.execute(query)
report = cursor.fetchall()
```

---

## 📖 Topic 3: Indexing & Transactions

```python
# 1. Creating Index for fast user_id lookups
cursor.execute("CREATE INDEX IF NOT EXISTS idx_expenses_user_id ON expenses(user_id);")

# 2. Atomic Bank Transaction (Rollback on failure)
def transfer_funds(conn, sender_id, receiver_id, amount):
    try:
        cur = conn.cursor()
        cur.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, sender_id))
        cur.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, receiver_id))
        conn.commit()  # Both operations succeed atomically
    except Exception:
        conn.rollback()  # Reverts both operations if error occurs
        raise
```

---

## ⚡ Master Cheat Sheet

```python
# Advanced SQL & Database Design Master Cheat Sheet

import sqlite3

# 1. Connection with Foreign Keys Enabled
conn = sqlite3.connect("app.db")
conn.execute("PRAGMA foreign_keys = ON;")

# 2. Complex Join with Group By & Having
sql = """
SELECT u.name, SUM(e.amount) AS total
FROM users u
LEFT JOIN expenses e ON u.id = e.user_id
GROUP BY u.id
HAVING total > 1000
ORDER BY total DESC
"""

# 3. Create Index
conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_user ON expenses(user_id);")
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `PRAGMA foreign_keys = ON;` in SQLite:**
   - ❌ Inserting invalid `user_id` values succeeds without error because SQLite disables foreign key checks by default.
   - ✅ Always execute `conn.execute("PRAGMA foreign_keys = ON;")` immediately after establishing connection.

2. **Using `WHERE` Instead of `HAVING` for Aggregate Conditions:**
   - ❌ `WHERE SUM(amount) > 1000` (Raises `OperationalError: misuse of aggregate function`).
   - ✅ Use `HAVING SUM(amount) > 1000` after `GROUP BY`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between `WHERE` and `HAVING` in SQL?
**Answer:** `WHERE` filters individual rows before grouping or aggregation takes place. `HAVING` filters aggregated group records after the `GROUP BY` operation has been computed.

### Q2: When should you create a Database Index?
**Answer:** Create indexes on columns that are frequently used in `WHERE` clauses, `JOIN` conditions (`FOREIGN KEY` columns), or `ORDER BY` statements on large tables to speed up search lookups. Avoid indexing write-heavy tables unnecessarily because indexes add disk and update overhead.

---

## 📝 Recap Checklist
- [x] Designed relational database schemas with Primary and Foreign Keys.
- [x] Enabled foreign key constraint checks in SQLite (`PRAGMA foreign_keys = ON;`).
- [x] Wrote `INNER JOIN` and `LEFT JOIN` queries.
- [x] Grouped data using `GROUP BY` and filtered aggregates with `HAVING`.
- [x] Built database B-Tree indexes (`CREATE INDEX`).
- [x] Implemented atomic transactions with `commit()` and `rollback()`.
- [x] Built Multi-User Expense Tracker and Monthly Reporting System projects.

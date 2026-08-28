# ==============================================================================
# Program    : Advanced Relational SQLite Practice (Tasks 1 to 8)
# Objective  : Implement relational database schema, Foreign Keys, JOINs, GROUP BY, HAVING, and Indexes.
# Concept    : Relational Database Design, SQL JOINs, Aggregations & Indexing
# Why Used   : Complete walkthrough covering all 8 practice tasks in Day 26 requirements.
# ==============================================================================

import os
import sqlite3

DB_FILE = "advanced_practice.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    # What is used : PRAGMA foreign_keys = ON;
    # Why it is used: Mandatory in SQLite to enforce foreign key constraints across tables
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_relational_schema(conn: sqlite3.Connection) -> None:
    """Task 1: Create users, categories, expenses tables with Primary and Foreign Keys."""
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)

    # 2. Categories Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    # 3. Expenses Table (referencing users and categories)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    print("[Task 1] Relational schema (users, categories, expenses) created successfully.")

def populate_sample_data(conn: sqlite3.Connection) -> None:
    """Task 2: Insert 5 users, 5 categories, and 20 expenses."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expenses;")
    cursor.execute("DELETE FROM categories;")
    cursor.execute("DELETE FROM users;")

    users = [
        ("Suraj Sawant", "suraj@example.com"),
        ("Rahul Sharma", "rahul@example.com"),
        ("Aditi Patel", "aditi@example.com"),
        ("Priya Verma", "priya@example.com"),
        ("Amit Kumar", "amit@example.com")
    ]
    cursor.executemany("INSERT INTO users (name, email) VALUES (?, ?)", users)

    categories = [
        ("Food",),
        ("Travel",),
        ("Shopping",),
        ("Entertainment",),
        ("Bills",)
    ]
    cursor.executemany("INSERT INTO categories (name) VALUES (?)", categories)

    expenses = [
        (1, 1, 450.0, "Lunch", "2026-08-01"),
        (1, 1, 2000.0, "Dinner Party", "2026-08-02"),
        (1, 2, 1200.0, "Train Pass", "2026-08-03"),
        (1, 3, 3100.0, "Clothes", "2026-08-04"),
        (1, 4, 800.0, "Movie Ticket", "2026-08-05"),
        (2, 1, 350.0, "Breakfast", "2026-08-01"),
        (2, 2, 1500.0, "Taxi", "2026-08-02"),
        (2, 5, 2200.0, "Electricity Bill", "2026-08-03"),
        (2, 3, 600.0, "Books", "2026-08-04"),
        (3, 1, 1200.0, "Restaurant", "2026-08-01"),
        (3, 3, 4500.0, "Smartphone", "2026-08-02"),
        (3, 4, 1500.0, "Concert Ticket", "2026-08-03"),
        (3, 5, 1900.0, "Internet Bill", "2026-08-04"),
        (4, 1, 500.0, "Snacks", "2026-08-01"),
        (4, 2, 800.0, "Bus Fare", "2026-08-02"),
        (4, 5, 1100.0, "Water Bill", "2026-08-03"),
        (5, 1, 300.0, "Coffee", "2026-08-01"),
        (5, 2, 400.0, "Metro Ride", "2026-08-02"),
        (5, 3, 1200.0, "Shoes", "2026-08-03"),
        (5, 4, 600.0, "Gaming Sub", "2026-08-04")
    ]
    cursor.executemany("INSERT INTO expenses (user_id, category_id, amount, description, date) VALUES (?, ?, ?, ?, ?)", expenses)
    conn.commit()
    print("[Task 2] Inserted 5 users, 5 categories, and 20 expenses.")

def display_joined_expenses(conn: sqlite3.Connection) -> None:
    """Task 3: Display User Name | Category | Description | Amount using INNER JOIN."""
    cursor = conn.cursor()
    query = """
        SELECT users.name, categories.name, expenses.description, expenses.amount
        FROM expenses
        INNER JOIN users ON expenses.user_id = users.id
        INNER JOIN categories ON expenses.category_id = categories.id
        LIMIT 5
    """
    cursor.execute(query)
    records = cursor.fetchall()
    print("\n--- [Task 3] DISPLAY USER EXPENSES USING JOIN (SAMPLE 5) ---")
    for u_name, c_name, desc, amt in records:
        print(f"User: {u_name:<15} | Category: {c_name:<12} | Desc: {desc:<15} | Amount: Rs.{amt:.2f}")

def total_spending_per_user(conn: sqlite3.Connection) -> None:
    """Task 4: Calculate total spending per user using JOIN and GROUP BY."""
    cursor = conn.cursor()
    query = """
        SELECT users.name, SUM(expenses.amount) AS total_spent
        FROM users
        INNER JOIN expenses ON users.id = expenses.user_id
        GROUP BY users.id
        ORDER BY total_spent DESC
    """
    cursor.execute(query)
    records = cursor.fetchall()
    print("\n--- [Task 4] TOTAL SPENDING PER USER (GROUP BY USERS) ---")
    for name, total in records:
        print(f"User: {name:<16} | Total Spent: Rs.{total:,.2f}")

def total_spending_per_category(conn: sqlite3.Connection) -> None:
    """Task 5: Calculate spending per category using JOIN and GROUP BY."""
    cursor = conn.cursor()
    query = """
        SELECT categories.name, SUM(expenses.amount) AS total_spent
        FROM categories
        INNER JOIN expenses ON categories.id = expenses.category_id
        GROUP BY categories.id
        ORDER BY total_spent DESC
    """
    cursor.execute(query)
    records = cursor.fetchall()
    print("\n--- [Task 5] TOTAL SPENDING PER CATEGORY ---")
    for cat_name, total in records:
        print(f"Category: {cat_name:<14} | Total Spent: Rs.{total:,.2f}")

def categories_exceeding_threshold(conn: sqlite3.Connection, threshold: float = 1000.0) -> None:
    """Task 6: Find categories where total spending exceeds threshold using HAVING."""
    cursor = conn.cursor()
    # What is used : GROUP BY with HAVING SUM(amount) > threshold
    # Why it is used: Filters aggregated group totals after computing sums
    query = """
        SELECT categories.name, SUM(expenses.amount) AS total_spent
        FROM categories
        INNER JOIN expenses ON categories.id = expenses.category_id
        GROUP BY categories.id
        HAVING SUM(expenses.amount) > ?
        ORDER BY total_spent DESC
    """
    cursor.execute(query, (threshold,))
    records = cursor.fetchall()
    print(f"\n--- [Task 6] CATEGORIES SPENDING EXCEEDING Rs.{threshold:,.2f} (HAVING) ---")
    for cat_name, total in records:
        print(f"Category: {cat_name:<14} | Total Spent: Rs.{total:,.2f}")

def highest_spending_user(conn: sqlite3.Connection) -> None:
    """Task 7: Find the user with the highest total spending."""
    cursor = conn.cursor()
    query = """
        SELECT users.name, SUM(expenses.amount) AS total_spent
        FROM users
        INNER JOIN expenses ON users.id = expenses.user_id
        GROUP BY users.id
        ORDER BY total_spent DESC
        LIMIT 1
    """
    cursor.execute(query)
    record = cursor.fetchone()
    print("\n--- [Task 7] HIGHEST SPENDING USER ---")
    if record:
        print(f"Top Spender: {record[0]} (Total: Rs.{record[1]:,.2f})")

def create_indexes(conn: sqlite3.Connection) -> None:
    """Task 8: Create an index on expenses.user_id."""
    cursor = conn.cursor()
    # What is used : CREATE INDEX
    # Why it is used: Speeds up queries filtering or joining on expenses.user_id
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_expenses_user_id ON expenses(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_expenses_category_id ON expenses(category_id);")
    conn.commit()
    print("\n[Task 8] Created indexes on expenses(user_id) and expenses(category_id) successfully.")

def main() -> None:
    print("=== DAY 26: ADVANCED SQL & DATABASE DESIGN PRACTICE TASKS ===")
    conn = get_connection()
    try:
        init_relational_schema(conn)
        populate_sample_data(conn)
        display_joined_expenses(conn)
        total_spending_per_user(conn)
        total_spending_per_category(conn)
        categories_exceeding_threshold(conn, 1000.0)
        highest_spending_user(conn)
        create_indexes(conn)
    finally:
        conn.close()

    # Cleanup test db file
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except OSError:
            pass

if __name__ == "__main__":
    main()

# ==============================================================================
# Program    : Expense Service (Business Logic Layer)
# Objective  : Handles validation and business operations for Users, Categories, and Expenses.
# Concept    : Service Layer & Dependency Injection
# Why Used   : Receives Database instance via constructor dependency injection.
# ==============================================================================

from datetime import datetime
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database

class ExpenseService:
    def __init__(self, database: Database):
        # What is used : Dependency Injection (self.database = database)
        # Why it is used: Injects database dependency allowing mock database replacement in unit tests
        self.database = database

    def add_user(self, name: str, email: str) -> int:
        if not name or not email:
            raise ValueError("Name and email are required.")
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
            conn.commit()
            return cursor.lastrowid

    def add_category(self, name: str) -> int:
        if not name:
            raise ValueError("Category name is required.")
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO categories (name) VALUES (?)", (name,))
            conn.commit()
            return cursor.lastrowid

    def add_expense(self, user_id: int, category_id: int, amount: float, description: str = "") -> int:
        if amount <= 0:
            raise ValueError("Amount must be greater than zero.")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (user_id, category_id, amount, description, date) VALUES (?, ?, ?, ?, ?)",
                (user_id, category_id, amount, description, now_str)
            )
            conn.commit()
            return cursor.lastrowid

    def get_user_expenses(self, user_id: int) -> list[tuple]:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT e.id, u.name, c.name, e.amount, e.description, e.date
                FROM expenses e
                INNER JOIN users u ON e.user_id = u.id
                INNER JOIN categories c ON e.category_id = c.id
                WHERE e.user_id = ?
                ORDER BY e.id DESC
            """
            cursor.execute(query, (user_id,))
            return cursor.fetchall()

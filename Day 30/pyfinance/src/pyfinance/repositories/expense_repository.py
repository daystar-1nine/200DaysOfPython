# ==============================================================================
# Program    : Expense Repository (Data Access Layer)
# Objective  : Provide CRUD and search database operations for Expense objects.
# Concept    : Repository Design Pattern
# Why Used   : Isolates SQL query details from business logic services.
# ==============================================================================

import os
import sqlite3
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.database import Database
from pyfinance.models.expense import Expense, Budget

class ExpenseRepository:
    def __init__(self, database: Database):
        self.database = database

    def add(self, expense: Expense) -> int:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, description, date, created_at) VALUES (?, ?, ?, ?, ?)",
                (expense.amount, expense.category, expense.description, expense.date, expense.created_at)
            )
            conn.commit()
            return cursor.lastrowid

    def get_by_id(self, expense_id: int) -> Expense | None:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, amount, category, description, date, created_at FROM expenses WHERE id = ?", (expense_id,))
            row = cursor.fetchone()
            if row:
                return Expense(id=row[0], amount=row[1], category=row[2], description=row[3], date=row[4], created_at=row[5])
            return None

    def get_all(self) -> list[Expense]:
        return self.get_all_records()

    def get_all_records(self) -> list[Expense]:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, amount, category, description, date, created_at FROM expenses ORDER BY date DESC, id DESC")
            rows = cursor.fetchall()
            return [Expense(id=r[0], amount=r[1], category=r[2], description=r[3], date=r[4], created_at=r[5]) for r in rows]

    def update(self, expense_id: int, amount: float | None = None, category: str | None = None, description: str | None = None, date: str | None = None) -> bool:
        current = self.get_by_id(expense_id)
        if not current:
            return False

        new_amount = amount if amount is not None else current.amount
        new_category = category if category is not None else current.category
        new_desc = description if description is not None else current.description
        new_date = date if date is not None else current.date

        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE expenses SET amount = ?, category = ?, description = ?, date = ? WHERE id = ?",
                (new_amount, new_category, new_desc, new_date, expense_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, expense_id: int) -> bool:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
            conn.commit()
            return cursor.rowcount > 0

    def search(self, category: str | None = None, start_date: str | None = None, end_date: str | None = None, keyword: str | None = None) -> list[Expense]:
        query = "SELECT id, amount, category, description, date, created_at FROM expenses WHERE 1=1"
        params = []

        if category:
            query += " AND category LIKE ?"
            params.append(f"%{category}%")
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if keyword:
            query += " AND (description LIKE ? OR category LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        query += " ORDER BY date DESC, id DESC"

        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [Expense(id=r[0], amount=r[1], category=r[2], description=r[3], date=r[4], created_at=r[5]) for r in rows]

    def set_budget(self, category: str, limit: float) -> None:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO budgets (category, monthly_limit) VALUES (?, ?) ON CONFLICT(category) DO UPDATE SET monthly_limit = ?",
                (category, limit, limit)
            )
            conn.commit()

    def get_budgets(self) -> list[Budget]:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, category, monthly_limit FROM budgets")
            rows = cursor.fetchall()
            return [Budget(id=r[0], category=r[1], monthly_limit=r[2]) for r in rows]

# ==============================================================================
# Program    : PyFinance Database Access Layer
# Objective  : SQLite database connection management and relational schema initialization.
# Concept    : Relational Database Access & Foreign Keys
# Why Used   : Creates expenses and budgets tables with indexes.
# ==============================================================================

import os
import sqlite3

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Expenses Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL CHECK(amount > 0),
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # Budgets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT UNIQUE NOT NULL,
                    monthly_limit REAL NOT NULL CHECK(monthly_limit > 0)
                )
            """)
            # Indexes for fast query lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);")
            conn.commit()

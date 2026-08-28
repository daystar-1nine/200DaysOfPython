# ==============================================================================
# Test Suite : Database Layer Integration Tests
# Objective  : Verify SQLite table initialization, foreign key enforcement, and indexes.
# Concept    : Database Integration Testing
# Why Used   : Asserts relational schema integrity.
# ==============================================================================

import os
import sqlite3
import sys
import unittest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 28", "expense_tracker", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database

def test_database_table_creation(database):
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        assert "users" in tables
        assert "categories" in tables
        assert "expenses" in tables

def test_foreign_key_enforcement(database):
    with database.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO expenses (user_id, category_id, amount, date) VALUES (999, 999, 100.0, '2026-08-28')", ())
            conn.commit()
            assert False, "Should have raised IntegrityError for invalid FK"
        except sqlite3.IntegrityError:
            assert True

class TestDatabaseIntegrationRunner(unittest.TestCase):
    def test_database_standalone(self):
        db_file = "temp_int_db.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [t[0] for t in cursor.fetchall()]
                self.assertIn("users", tables)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

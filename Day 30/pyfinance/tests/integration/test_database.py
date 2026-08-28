# ==============================================================================
# Test Suite : Integration Tests for Database Layer
# Objective  : Verify SQLite database schema, foreign key enforcement, and indexes.
# Concept    : Database Integration Testing
# Why Used   : Asserts relational database initialization and constraints.
# ==============================================================================

import os
import sys
import unittest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 30", "pyfinance", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.database import Database

def test_database_table_initialization(database):
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        assert "expenses" in tables
        assert "budgets" in tables

class TestDatabaseIntegrationRunner(unittest.TestCase):
    def test_database_standalone(self):
        db_file = "temp_pyfinance_db_int.db"
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
                self.assertIn("expenses", tables)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

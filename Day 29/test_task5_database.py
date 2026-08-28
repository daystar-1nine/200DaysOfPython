# ==============================================================================
# Test Suite : Task 5 Database CRUD Integration Pytest Suite
# Objective  : Test Create, Read, Update, Delete operations using isolated test database.
# Concept    : Isolated Database Testing with Fixtures
# Why Used   : Ensures database operations work without altering production databases.
# ==============================================================================

import os
import sqlite3
import sys
import unittest
import pytest

DB_TEST_FILE = "task5_temp_test.db"

@pytest.fixture
def db_conn():
    if os.path.exists(DB_TEST_FILE):
        try:
            os.remove(DB_TEST_FILE)
        except OSError:
            pass
    conn = sqlite3.connect(DB_TEST_FILE)
    conn.execute("CREATE TABLE expenses (id INTEGER PRIMARY KEY, category TEXT, amount REAL);")
    conn.commit()
    yield conn
    conn.close()
    if os.path.exists(DB_TEST_FILE):
        try:
            os.remove(DB_TEST_FILE)
        except OSError:
            pass

def test_database_crud(db_conn):
    cursor = db_conn.cursor()
    
    # 1. Create
    cursor.execute("INSERT INTO expenses (category, amount) VALUES (?, ?)", ("Food", 250.0))
    db_conn.commit()
    eid = cursor.lastrowid
    assert eid > 0

    # 2. Read
    cursor.execute("SELECT category, amount FROM expenses WHERE id = ?", (eid,))
    row = cursor.fetchone()
    assert row == ("Food", 250.0)

    # 3. Update
    cursor.execute("UPDATE expenses SET amount = ? WHERE id = ?", (300.0, eid))
    db_conn.commit()
    cursor.execute("SELECT amount FROM expenses WHERE id = ?", (eid,))
    assert cursor.fetchone()[0] == 300.0

    # 4. Delete
    cursor.execute("DELETE FROM expenses WHERE id = ?", (eid,))
    db_conn.commit()
    cursor.execute("SELECT COUNT(*) FROM expenses WHERE id = ?", (eid,))
    assert cursor.fetchone()[0] == 0

class TestTask5Runner(unittest.TestCase):
    def test_database_crud_standalone(self):
        if os.path.exists(DB_TEST_FILE):
            try:
                os.remove(DB_TEST_FILE)
            except OSError:
                pass
        conn = sqlite3.connect(DB_TEST_FILE)
        conn.execute("CREATE TABLE expenses (id INTEGER PRIMARY KEY, category TEXT, amount REAL);")
        conn.commit()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO expenses (category, amount) VALUES (?, ?)", ("Food", 250.0))
            conn.commit()
            eid = cursor.lastrowid
            self.assertGreater(eid, 0)

            cursor.execute("SELECT category, amount FROM expenses WHERE id = ?", (eid,))
            self.assertEqual(cursor.fetchone(), ("Food", 250.0))

            cursor.execute("UPDATE expenses SET amount = ? WHERE id = ?", (300.0, eid))
            conn.commit()
            cursor.execute("SELECT amount FROM expenses WHERE id = ?", (eid,))
            self.assertEqual(cursor.fetchone()[0], 300.0)

            cursor.execute("DELETE FROM expenses WHERE id = ?", (eid,))
            conn.commit()
            cursor.execute("SELECT COUNT(*) FROM expenses WHERE id = ?", (eid,))
            self.assertEqual(cursor.fetchone()[0], 0)
        finally:
            conn.close()
            if os.path.exists(DB_TEST_FILE):
                try:
                    os.remove(DB_TEST_FILE)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

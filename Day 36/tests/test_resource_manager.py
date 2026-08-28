# ==============================================================================
# Test Suite : Day 36 Resource Manager Pytest Suite
# Objective  : Test normal execution, exception propagation, cleanup, commit, rollback, and nested contexts.
# Concept    : Unit Testing Context Managers & Resource Cleanup
# Why Used   : Asserts resource safety, transaction integrity, and cleanup guarantees.
# ==============================================================================

import os
import sqlite3
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource_manager"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from database_manager import DatabaseManager, transaction
from timer_manager import TimerManager, execution_timer
from temp_file_manager import TemporaryFileManager, temp_file

# --- Database & Transaction Tests ---
def test_database_manager_commit(tmp_path):
    db_file = str(tmp_path / "test_commit.db")
    with DatabaseManager(db_file) as conn:
        conn.execute("CREATE TABLE items (id INT, name TEXT);")
        conn.execute("INSERT INTO items VALUES (1, 'Book');")
    
    # Verify commit persisted
    conn2 = sqlite3.connect(db_file)
    cursor = conn2.cursor()
    cursor.execute("SELECT name FROM items;")
    assert cursor.fetchone()[0] == 'Book'
    conn2.close()

def test_database_manager_rollback_on_exception(tmp_path):
    db_file = str(tmp_path / "test_rollback.db")
    # First create table
    with DatabaseManager(db_file) as conn:
        conn.execute("CREATE TABLE items (id INT, name TEXT);")
    
    # Execute failing block
    with pytest.raises(ValueError, match="Tx Failure"):
        with DatabaseManager(db_file) as conn:
            conn.execute("INSERT INTO items VALUES (1, 'Unsaved');")
            raise ValueError("Tx Failure")

    # Verify rollback prevented persistence
    conn2 = sqlite3.connect(db_file)
    cursor = conn2.cursor()
    cursor.execute("SELECT COUNT(*) FROM items;")
    assert cursor.fetchone()[0] == 0
    conn2.close()

# --- Temporary File Tests ---
def test_temp_file_manager_auto_cleanup():
    created_path = None
    with TemporaryFileManager(content="Scratch Data") as tmp_path:
        created_path = tmp_path
        assert os.path.exists(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            assert f.read() == "Scratch Data"

    # Assert auto-cleaned after with block exit
    assert not os.path.exists(created_path)

def test_temp_file_generator_cleanup():
    created_path = None
    with temp_file(content="Gen Content") as tmp_path:
        created_path = tmp_path
        assert os.path.exists(tmp_path)

    assert not os.path.exists(created_path)

# --- Timer Tests ---
def test_timer_manager_elapsed():
    with TimerManager("Test Block") as t:
        _ = sum(range(1000))
    assert t.elapsed > 0.0

# --- Nested Contexts Test ---
def test_nested_contexts(tmp_path):
    db_file = str(tmp_path / "nested.db")
    with TimerManager("Nested Execution"):
        with DatabaseManager(db_file) as conn:
            with TemporaryFileManager(content="Temp SQL Data") as tmp_path_str:
                conn.execute("CREATE TABLE logs (msg TEXT);")
                with open(tmp_path_str, "r", encoding="utf-8") as f:
                    conn.execute("INSERT INTO logs VALUES (?);", (f.read(),))

    conn2 = sqlite3.connect(db_file)
    assert conn2.execute("SELECT msg FROM logs;").fetchone()[0] == "Temp SQL Data"
    conn2.close()

class TestResourceManagerRunner(unittest.TestCase):
    def test_resource_manager_standalone(self):
        with TimerManager("Standalone Timer") as t:
            self.assertIsNotNone(t)

if __name__ == "__main__":
    unittest.main()

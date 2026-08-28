# ==============================================================================
# Program    : Database & Transaction Context Managers (database_manager.py)
# Objective  : Provide SQLite database connection and transaction context managers (COMMIT/ROLLBACK).
# Concept    : Context Manager Atomicity Pattern
# Why Used   : Guarantees database transaction commits on success or rollback on error.
# ==============================================================================

from contextlib import contextmanager
import sqlite3

class DatabaseManager:
    """Class-based database connection and transaction context manager."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self) -> sqlite3.Connection:
        # What is used : Connection Acquisition & Foreign Keys
        # Why it is used: Opens connection and begins transaction boundary
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.conn:
            if exc_type is not None:
                print(f"[DB] Error encountered ({exc_val}): Executing ROLLBACK...")
                self.conn.rollback()
            else:
                print("[DB] Operation succeeded: Executing COMMIT...")
                self.conn.commit()
            self.conn.close()
        # Return False to propagate exceptions to caller
        return False

@contextmanager
def transaction(conn: sqlite3.Connection):
    """Generator-based transaction context manager for existing connection."""
    print("[TX] Transaction started...")
    try:
        yield conn
        print("[TX] Transaction success: Executing COMMIT...")
        conn.commit()
    except Exception as e:
        print(f"[TX] Transaction failed ({e}): Executing ROLLBACK...")
        conn.rollback()
        raise e

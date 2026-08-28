# ==============================================================================
# Program    : Application Service Layer with Exception Chaining (app_service.py)
# Objective  : Demonstrate converting low-level errors into domain exceptions via chaining.
# Concept    : Exception Chaining (raise DomainError from cause)
# Why Used   : Preserves original cause stack trace while providing user-friendly errors.
# ==============================================================================

import os
import sqlite3
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from exceptions import (
    ApplicationError,
    ValidationError,
    BoundsError,
    DatabaseError,
    UniqueConstraintError,
    NotFoundError,
    AuthenticationError
)

class UserService:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._init_db()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_file)
            conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE, age INT);")
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            # What is used : Explicit Exception Chaining
            # Why it is used: Preserves low-level sqlite3 traceback in __cause__ attribute
            raise DatabaseError("Failed initializing database schema.") from e

    def register_user(self, email: str, age: int) -> int:
        if not email or "@" not in email:
            raise ValidationError(f"Invalid email format: '{email}'")
        if age < 18 or age > 120:
            raise BoundsError(f"Age {age} is out of permitted registration bounds (18-120).")

        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (email, age) VALUES (?, ?)", (email, age))
            conn.commit()
            uid = cursor.lastrowid
            conn.close()
            return uid
        except sqlite3.IntegrityError as e:
            raise UniqueConstraintError(f"User with email '{email}' already exists.") from e
        except sqlite3.Error as e:
            raise DatabaseError(f"Database error registering user '{email}'.") from e

    def get_user_by_id(self, user_id: int) -> dict:
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id, email, age FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                raise NotFoundError(f"User with ID #{user_id} was not found.")
            return {"id": row[0], "email": row[1], "age": row[2]}
        except sqlite3.Error as e:
            raise DatabaseError(f"Query error fetching user ID #{user_id}.") from e


if __name__ == "__main__":
    print("=== ADVANCED EXCEPTION HANDLER DEMO ===")
    service = UserService("demo_users.db")
    try:
        service.register_user("invalid_email", 25)
    except ApplicationError as err:
        print(f"Caught Application Domain Error: {err}")
        if err.__cause__:
            print(f"  Root Cause Exception: {repr(err.__cause__)}")

    if os.path.exists("demo_users.db"):
        try:
            os.remove("demo_users.db")
        except OSError:
            pass

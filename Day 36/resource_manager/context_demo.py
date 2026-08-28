# ==============================================================================
# Program    : Context Managers Interactive Demo (context_demo.py)
# Objective  : Demonstrate Class & Generator context managers in action.
# Concept    : Context Manager Execution
# Why Used   : Interactive demonstration of Database, Timer, and TempFile context managers.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from database_manager import DatabaseManager, transaction
from timer_manager import TimerManager, execution_timer
from temp_file_manager import TemporaryFileManager, temp_file

def main():
    print("==================================================")
    print("      DAY 36 - RESOURCE MANAGER CONTEXT DEMO      ")
    print("==================================================\n")

    db_path = os.path.join(os.path.dirname(__file__), "demo.db")

    print("--- 1. Database Success Transaction (COMMIT) ---")
    with DatabaseManager(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INT, name TEXT);")
        conn.execute("INSERT INTO users VALUES (1, 'Suraj');")
    print()

    print("--- 2. Database Failed Transaction (ROLLBACK) ---")
    try:
        with DatabaseManager(db_path) as conn:
            conn.execute("INSERT INTO users VALUES (2, 'Aniket');")
            raise ValueError("Simulated unexpected failure during insertion!")
    except ValueError as err:
        print(f"Caught Error: {err}\n")

    print("--- 3. Timer & Temp File Context Managers ---")
    with TimerManager("File Operation") as t:
        with TemporaryFileManager(content="Hello Context Managers!") as tmp_path:
            with open(tmp_path, "r", encoding="utf-8") as f:
                print(f"Read Content: '{f.read()}'")

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass

if __name__ == "__main__":
    main()

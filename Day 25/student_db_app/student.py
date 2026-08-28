# ==============================================================================
# Program    : CLI Student Database Application (Mini Project)
# Objective  : CLI application managing student records in SQLite database 'students.db'.
# Concept    : CLI Subcommands (argparse) + SQLite CRUD Operations
# Why Used   : Connects CLI interface to persistent SQLite database for full student CRUD operations.
# ==============================================================================

import argparse
import logging
import os
import sqlite3
import sys

DB_FILE = os.path.join(os.path.dirname(__file__), "students.db")
LOG_FILE = os.path.join(os.path.dirname(__file__), "student_app.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def init_db() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                marks REAL
            )
        """)
        conn.commit()

def handle_add(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, age, marks) VALUES (?, ?, ?)",
            (args.name, args.age, args.marks)
        )
        conn.commit()
        new_id = cursor.lastrowid
        logging.info("Added Student ID %d: %s, Age: %d, Marks: %.2f", new_id, args.name, args.age, args.marks)
        print(f"[SUCCESS] Added Student ID #{new_id}: {args.name} (Age: {args.age}, Marks: {args.marks})")

def handle_list(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, marks FROM students ORDER BY id ASC")
        records = cursor.fetchall()
        if not records:
            print("No student records found.")
            return
        print("\n------------------ STUDENT DATABASE LIST ------------------")
        for r in records:
            print(f"ID: {r[0]:<4} | Name: {r[1]:<18} | Age: {r[2]:<3} | Marks: {r[3]:.2f}")
        print("-----------------------------------------------------------\n")

def handle_search(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, marks FROM students WHERE id = ?", (args.id,))
        record = cursor.fetchone()
        if record:
            print(f"\n[FOUND] Student ID #{record[0]}: Name: {record[1]}, Age: {record[2]}, Marks: {record[3]}\n")
        else:
            print(f"[NOT FOUND] Student ID #{args.id} does not exist.")

def handle_update(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE students SET marks = ? WHERE id = ?", (args.marks, args.id))
        if cursor.rowcount > 0:
            conn.commit()
            logging.info("Updated Student ID %d marks to %.2f", args.id, args.marks)
            print(f"[SUCCESS] Updated Student ID #{args.id} marks to {args.marks:.2f}")
        else:
            print(f"[WARNING] Student ID #{args.id} not found.")

def handle_delete(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (args.id,))
        if cursor.rowcount > 0:
            conn.commit()
            logging.info("Deleted Student ID %d", args.id)
            print(f"[SUCCESS] Deleted Student ID #{args.id}")
        else:
            print(f"[WARNING] Student ID #{args.id} not found.")

def handle_stats(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(marks), MAX(marks), MIN(marks) FROM students")
        row = cursor.fetchone()
        total, avg_m, max_m, min_m = row
        print("\n------ STUDENT DATABASE STATISTICS ------")
        print(f"Total Students : {total}")
        print(f"Average Marks  : {avg_m:.2f}" if avg_m else "Average Marks  : N/A")
        print(f"Highest Marks  : {max_m:.2f}" if max_m else "Highest Marks  : N/A")
        print(f"Lowest Marks   : {min_m:.2f}" if min_m else "Lowest Marks   : N/A")
        print("-----------------------------------------\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Student Database Application")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Add
    add_p = subparsers.add_parser("add", help="Add new student")
    add_p.add_argument("--name", type=str, required=True, help="Student name")
    add_p.add_argument("--age", type=int, default=20, help="Student age")
    add_p.add_argument("--marks", type=float, required=True, help="Student marks")

    # List
    list_p = subparsers.add_parser("list", help="List all students")

    # Search
    search_p = subparsers.add_parser("search", help="Search student by ID")
    search_p.add_argument("--id", type=int, required=True, help="Student ID")

    # Update
    update_p = subparsers.add_parser("update", help="Update student marks")
    update_p.add_argument("--id", type=int, required=True, help="Student ID")
    update_p.add_argument("--marks", type=float, required=True, help="New marks value")

    # Delete
    delete_p = subparsers.add_parser("delete", help="Delete student by ID")
    delete_p.add_argument("--id", type=int, required=True, help="Student ID")

    # Stats
    stats_p = subparsers.add_parser("stats", help="Display summary statistics")

    return parser

def main() -> None:
    print("=== MINI PROJECT: CLI STUDENT DATABASE APP ===")
    init_db()
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        handle_add(parser.parse_args(["add", "--name", "Suraj Sawant", "--age", "20", "--marks", "88.5"]))
        handle_add(parser.parse_args(["add", "--name", "Neha Verma", "--age", "21", "--marks", "95.0"]))
        handle_list(parser.parse_args(["list"]))
        handle_search(parser.parse_args(["search", "--id", "1"]))
        handle_update(parser.parse_args(["update", "--id", "1", "--marks", "92.0"]))
        handle_stats(parser.parse_args(["stats"]))
    else:
        args = parser.parse_args()
        if args.command == "add":
            handle_add(args)
        elif args.command == "list":
            handle_list(args)
        elif args.command == "search":
            handle_search(args)
        elif args.command == "update":
            handle_update(args)
        elif args.command == "delete":
            handle_delete(args)
        elif args.command == "stats":
            handle_stats(args)

if __name__ == "__main__":
    main()

# ==============================================================================
# Program    : Database Expense Tracker App (Challenge Project)
# Objective  : Upgrade CLI Expense Tracker to use SQLite database 'expenses.db'.
# Concept    : SQLite Persistence, Aggregates (SUM, GROUP BY) & Subcommands
# Why Used   : Uses SQLite table expenses(id, category, amount, description, date) for persistent tracking.
# ==============================================================================

import argparse
from datetime import datetime
import logging
import os
import sqlite3
import sys

DB_FILE = os.path.join(os.path.dirname(__file__), "expenses.db")
LOG_FILE = os.path.join(os.path.dirname(__file__), "expense_db.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def init_db() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                date TEXT NOT NULL
            )
        """)
        conn.commit()

def handle_add(args: argparse.Namespace) -> None:
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO expenses (category, amount, description, date) VALUES (?, ?, ?, ?)",
            (args.category, args.amount, args.description, current_date)
        )
        conn.commit()
        new_id = cursor.lastrowid
        logging.info("Added Expense ID %d: %s -> Rs.%.2f", new_id, args.category, args.amount)
        print(f"[SUCCESS] Added Expense #{new_id}: {args.category} -> Rs.{args.amount:,.2f} on {current_date}")

def handle_list(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, category, amount, description, date FROM expenses ORDER BY id DESC")
        records = cursor.fetchall()
        if not records:
            print("No expenses recorded in database.")
            return
        print("\n------------------ DATABASE EXPENSES LIST ------------------")
        for r in records:
            desc = f" ({r[3]})" if r[3] else ""
            print(f"ID: {r[0]:<4} | Category: {r[1]:<12} | Amount: Rs.{r[2]:<10.2f} | Date: {r[4]}{desc}")
        print("------------------------------------------------------------\n")

def handle_delete(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (args.id,))
        if cursor.rowcount > 0:
            conn.commit()
            logging.info("Deleted Expense ID %d", args.id)
            print(f"[SUCCESS] Deleted Expense ID #{args.id}")
        else:
            print(f"[WARNING] Expense ID #{args.id} not found.")

def handle_summary(args: argparse.Namespace) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT category, SUM(amount) FROM expenses GROUP BY category ORDER BY SUM(amount) DESC")
        category_totals = cursor.fetchall()

        cursor.execute("SELECT SUM(amount) FROM expenses")
        total_sum = cursor.fetchone()[0] or 0.0

        if not category_totals:
            print("No expenses recorded yet.")
            return

        print("\n------ DATABASE EXPENSE SUMMARY ------")
        for cat, amt in category_totals:
            print(f"{cat:<15} Rs.{amt:,.2f}")
        print("--------------------------------------")
        print(f"Total           Rs.{total_sum:,.2f}\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Database Expense Tracker")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Add
    add_p = subparsers.add_parser("add", help="Add new expense")
    add_p.add_argument("--category", type=str, required=True, help="Expense category")
    add_p.add_argument("--amount", type=float, required=True, help="Expense amount")
    add_p.add_argument("--description", type=str, default="", help="Optional description")

    # List
    list_p = subparsers.add_parser("list", help="List all expenses")

    # Delete
    delete_p = subparsers.add_parser("delete", help="Delete expense by ID")
    delete_p.add_argument("--id", type=int, required=True, help="Expense ID")

    # Summary
    summary_p = subparsers.add_parser("summary", help="Display summary report")

    return parser

def main() -> None:
    print("=== CHALLENGE PROJECT: DATABASE EXPENSE TRACKER ===")
    init_db()
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        handle_add(parser.parse_args(["add", "--category", "Food", "--amount", "2450.0", "--description", "Groceries"]))
        handle_add(parser.parse_args(["add", "--category", "Travel", "--amount", "1200.0", "--description", "Fuel"]))
        handle_add(parser.parse_args(["add", "--category", "Shopping", "--amount", "3100.0", "--description", "Clothes"]))
        handle_list(parser.parse_args(["list"]))
        handle_summary(parser.parse_args(["summary"]))
    else:
        args = parser.parse_args()
        if args.command == "add":
            handle_add(args)
        elif args.command == "list":
            handle_list(args)
        elif args.command == "delete":
            handle_delete(args)
        elif args.command == "summary":
            handle_summary(args)

if __name__ == "__main__":
    main()

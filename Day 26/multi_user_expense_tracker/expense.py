# ==============================================================================
# Program    : Multi-User Relational Expense Tracker (Mini Project)
# Objective  : Build multi-table CLI expense application (users, categories, expenses).
# Concept    : Relational Database Design, Foreign Keys, Subcommands & Joins
# Why Used   : Supports add-user, add-category, add-expense, list, and summary per user.
# ==============================================================================

import argparse
from datetime import datetime
import logging
import os
import sqlite3
import sys

DB_FILE = os.path.join(os.path.dirname(__file__), "relational_expenses.db")
LOG_FILE = os.path.join(os.path.dirname(__file__), "relational_expense.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db() -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (category_id) REFERENCES categories(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expenses_user ON expenses(user_id);")
        conn.commit()

def handle_add_user(name: str, email: str) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
            conn.commit()
            uid = cursor.lastrowid
            logging.info("Added User ID %d: %s (%s)", uid, name, email)
            print(f"[SUCCESS] Added User #{uid}: {name} ({email})")
        except sqlite3.IntegrityError:
            print(f"[ERROR] User with email '{email}' already exists.")

def handle_add_category(name: str) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO categories (name) VALUES (?)", (name,))
            conn.commit()
            cid = cursor.lastrowid
            logging.info("Added Category ID %d: %s", cid, name)
            print(f"[SUCCESS] Added Category #{cid}: {name}")
        except sqlite3.IntegrityError:
            print(f"[ERROR] Category '{name}' already exists.")

def handle_add_expense(user_id: int, category_id: int, amount: float, description: str = "") -> None:
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO expenses (user_id, category_id, amount, description, date) VALUES (?, ?, ?, ?, ?)",
                (user_id, category_id, amount, description, current_date)
            )
            conn.commit()
            eid = cursor.lastrowid
            logging.info("Added Expense ID %d for User ID %d: Rs.%.2f", eid, user_id, amount)
            print(f"[SUCCESS] Added Expense #{eid}: User #{user_id}, Category #{category_id} -> Rs.{amount:,.2f}")
        except sqlite3.IntegrityError as e:
            print(f"[FOREIGN KEY ERROR] Invalid user_id (#{user_id}) or category_id (#{category_id}).")

def handle_list(user_id: int) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT expenses.id, users.name, categories.name, expenses.amount, expenses.description, expenses.date
            FROM expenses
            INNER JOIN users ON expenses.user_id = users.id
            INNER JOIN categories ON expenses.category_id = categories.id
            WHERE expenses.user_id = ?
            ORDER BY expenses.id DESC
        """
        cursor.execute(query, (user_id,))
        records = cursor.fetchall()
        if not records:
            print(f"No expenses found for User #{user_id}.")
            return
        user_name = records[0][1]
        print(f"\n------------------ EXPENSE LIST FOR {user_name.upper()} ------------------")
        for r in records:
            desc = f" ({r[4]})" if r[4] else ""
            print(f"ID: {r[0]:<4} | Category: {r[2]:<12} | Amount: Rs.{r[3]:<10.2f} | Date: {r[5]}{desc}")
        print("-----------------------------------------------------------------------\n")

def handle_summary(user_id: int) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        user_row = cursor.fetchone()
        if not user_row:
            print(f"[ERROR] User ID #{user_id} not found.")
            return
        user_name = user_row[0]

        query = """
            SELECT categories.name, SUM(expenses.amount)
            FROM expenses
            INNER JOIN categories ON expenses.category_id = categories.id
            WHERE expenses.user_id = ?
            GROUP BY categories.id
            ORDER BY SUM(expenses.amount) DESC
        """
        cursor.execute(query, (user_id,))
        cat_totals = cursor.fetchall()

        cursor.execute("SELECT SUM(amount) FROM expenses WHERE user_id = ?", (user_id,))
        total_sum = cursor.fetchone()[0] or 0.0

        print("\n+------------------------------------+")
        print("|          EXPENSE SUMMARY           |")
        print("+------------------------------------+")
        print(f"User: {user_name}\n")
        for cat, amt in cat_totals:
            print(f"{cat:<18} Rs.{amt:,.2f}")
        print("--------------------------------------")
        print(f"Total              Rs.{total_sum:,.2f}")
        print("+------------------------------------+\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-User Relational Expense Tracker")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # add-user
    au = subparsers.add_parser("add-user", help="Add new user")
    au.add_argument("--name", type=str, required=True, help="User name")
    au.add_argument("--email", type=str, required=True, help="User email")

    # add-category
    ac = subparsers.add_parser("add-category", help="Add new expense category")
    ac.add_argument("--name", type=str, required=True, help="Category name")

    # add-expense
    ae = subparsers.add_parser("add-expense", help="Add new expense record")
    ae.add_argument("--user", type=int, required=True, help="User ID")
    ae.add_argument("--category", type=int, required=True, help="Category ID")
    ae.add_argument("--amount", type=float, required=True, help="Expense amount")
    ae.add_argument("--description", type=str, default="", help="Description")

    # list
    ls = subparsers.add_parser("list", help="List user expenses")
    ls.add_argument("--user", type=int, required=True, help="User ID")

    # summary
    sm = subparsers.add_parser("summary", help="Summary user expense report")
    sm.add_argument("--user", type=int, required=True, help="User ID")

    return parser

def main() -> None:
    print("=== MINI PROJECT: MULTI-USER EXPENSE TRACKER ===")
    init_db()
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        handle_add_user("Suraj Sawant", "suraj@example.com")
        handle_add_category("Food")
        handle_add_category("Travel")
        handle_add_category("Shopping")
        handle_add_category("Entertainment")
        handle_add_expense(1, 1, 2450.0, "Monthly Groceries")
        handle_add_expense(1, 2, 1200.0, "Fuel Charges")
        handle_add_expense(1, 3, 3100.0, "Online Shopping")
        handle_add_expense(1, 4, 800.0, "Cinema Ticket")
        handle_list(1)
        handle_summary(1)
    else:
        args = parser.parse_args()
        if args.command == "add-user":
            handle_add_user(args.name, args.email)
        elif args.command == "add-category":
            handle_add_category(args.name)
        elif args.command == "add-expense":
            handle_add_expense(args.user, args.category, args.amount, args.description)
        elif args.command == "list":
            handle_list(args.user)
        elif args.command == "summary":
            handle_summary(args.user)

if __name__ == "__main__":
    main()

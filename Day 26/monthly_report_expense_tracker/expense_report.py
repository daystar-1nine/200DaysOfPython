# ==============================================================================
# Program    : Monthly Report Relational Expense Tracker (Challenge Project)
# Objective  : Generate detailed monthly category breakdowns and statistical expense analytics per user.
# Concept    : SQL Date Filtering, Group Aggregations & Complex Analytical Queries
# Why Used   : Adds `report --user ID --month MM --year YYYY` calculating category totals, highest category, highest expense, and transaction averages.
# ==============================================================================

import argparse
from datetime import datetime
import logging
import os
import sqlite3
import sys

DB_FILE = os.path.join(os.path.dirname(__file__), "monthly_expenses.db")
LOG_FILE = os.path.join(os.path.dirname(__file__), "monthly_expense.log")

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
        cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL);")
        cursor.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (category_id) REFERENCES categories(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_monthly_user_date ON expenses(user_id, date);")
        conn.commit()

def seed_demo_data() -> None:
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO users (name, email) VALUES ('Suraj Sawant', 'suraj@example.com')")
            cursor.executemany("INSERT INTO categories (name) VALUES (?)", [("Food",), ("Travel",), ("Shopping",), ("Bills",)])
            
            # Sample monthly expenses for August 2026
            records = [
                (1, 1, 3250.0, "Supermarket Groceries", "2026-08-05 10:30:00"),
                (1, 2, 1800.0, "Monthly Railway Pass", "2026-08-01 09:00:00"),
                (1, 3, 2400.0, "New Apparel", "2026-08-12 15:45:00"),
                (1, 1, 1200.0, "Weekend Dinner", "2026-08-20 20:15:00"),
                (1, 4, 2100.0, "Electricity & Gas", "2026-08-10 11:00:00")
            ]
            cursor.executemany("INSERT INTO expenses (user_id, category_id, amount, description, date) VALUES (?, ?, ?, ?, ?)", records)
            conn.commit()

def handle_monthly_report(user_id: int, month: str, year: str) -> None:
    # Ensure 2-digit month formatting (e.g. "08")
    month_str = f"{int(month):02d}"
    date_pattern = f"{year}-{month_str}%"

    with get_db() as conn:
        cursor = conn.cursor()
        
        # Verify user
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        u = cursor.fetchone()
        if not u:
            print(f"[ERROR] User ID #{user_id} not found.")
            return
        user_name = u[0]

        # Query Category Totals for month
        cat_query = """
            SELECT categories.name, SUM(expenses.amount)
            FROM expenses
            INNER JOIN categories ON expenses.category_id = categories.id
            WHERE expenses.user_id = ? AND expenses.date LIKE ?
            GROUP BY categories.id
            ORDER BY SUM(expenses.amount) DESC
        """
        cursor.execute(cat_query, (user_id, date_pattern))
        cat_totals = cursor.fetchall()

        if not cat_totals:
            print(f"No expense transactions found for {user_name} in {month_str}/{year}.")
            return

        # Detailed Metrics
        cursor.execute("SELECT SUM(amount), AVG(amount), MAX(amount), COUNT(*) FROM expenses WHERE user_id = ? AND date LIKE ?", (user_id, date_pattern))
        total_amt, avg_amt, max_amt, tx_count = cursor.fetchone()

        # Highest Single Expense Item
        cursor.execute("""
            SELECT expenses.description, expenses.amount, categories.name
            FROM expenses
            INNER JOIN categories ON expenses.category_id = categories.id
            WHERE expenses.user_id = ? AND expenses.date LIKE ?
            ORDER BY expenses.amount DESC LIMIT 1
        """, (user_id, date_pattern))
        top_item = cursor.fetchone()

        # Display Monthly Report
        print("\n==========================================================")
        print(f"       MONTHLY EXPENSE REPORT: {month_str}/{year}          ")
        print("==========================================================")
        print(f"User: {user_name}\n")
        print("--- CATEGORY BREAKDOWN ---")
        for cat, amt in cat_totals:
            print(f"{cat:<20} Rs.{amt:,.2f}")
        print("----------------------------------------------------------")
        print(f"Total Monthly Spending : Rs.{total_amt:,.2f}\n")

        print("--- STATISTICAL ANALYTICS ---")
        print(f"Highest Spending Category : {cat_totals[0][0]} (Rs.{cat_totals[0][1]:,.2f})")
        if top_item:
            print(f"Highest Single Expense    : '{top_item[0]}' [{top_item[2]}] -> Rs.{top_item[1]:,.2f}")
        print(f"Average Expense Amount    : Rs.{avg_amt:,.2f}")
        print(f"Total Transactions Count  : {tx_count}")
        print("==========================================================\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Monthly Expense Reporting Analytics Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # report subcommand
    rp = subparsers.add_parser("report", help="Generate monthly expense report")
    rp.add_argument("--user", type=int, required=True, help="User ID")
    rp.add_argument("--month", type=str, required=True, help="Month (e.g. 08 or 8)")
    rp.add_argument("--year", type=str, required=True, help="Year (e.g. 2026)")

    return parser

def main() -> None:
    print("=== ADVANCED CHALLENGE: MONTHLY REPORT EXPENSE TRACKER ===")
    seed_demo_data()
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand call:\n")
        handle_monthly_report(1, "08", "2026")
    else:
        args = parser.parse_args()
        if args.command == "report":
            handle_monthly_report(args.user, args.month, args.year)

if __name__ == "__main__":
    main()

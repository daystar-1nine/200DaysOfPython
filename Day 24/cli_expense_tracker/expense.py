# ==============================================================================
# Program    : CLI Expense Tracker App (Mini Project)
# Objective  : Build multi-command CLI expense tracker supporting add, list, delete, and summary commands.
# Concept    : argparse Subcommands (add_subparsers), JSON Persistence & Logging
# Why Used   : Integrates CLI argument routing with persistent storage, logging, and exception handling.
# ==============================================================================

import argparse
import json
import logging
import os
import sys

DATA_FILE = os.path.join(os.path.dirname(__file__), "expenses.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), "expense_cli.log")

# Configure logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def load_expenses() -> list[dict]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load expenses JSON: %s", e, exc_info=True)
        return []

def save_expenses(expenses: list[dict]) -> None:
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(expenses, f, indent=4)
    except Exception as e:
        logging.error("Failed to save expenses JSON: %s", e, exc_info=True)

def handle_add(args: argparse.Namespace) -> None:
    expenses = load_expenses()
    next_id = max([e.get("id", 0) for e in expenses], default=0) + 1
    new_entry = {
        "id": next_id,
        "category": args.category,
        "amount": args.amount
    }
    expenses.append(new_entry)
    save_expenses(expenses)
    logging.info("Added expense ID %d: %s -> Rs.%.2f", next_id, args.category, args.amount)
    print(f"[SUCCESS] Added Expense ID {next_id}: {args.category} -> Rs.{args.amount:,.2f}")

def handle_list(args: argparse.Namespace) -> None:
    expenses = load_expenses()
    if not expenses:
        print("No expenses recorded yet.")
        return
    print("\n------------------ EXPENSE LIST ------------------")
    for e in expenses:
        print(f"ID: {e['id']:<4} | Category: {e['category']:<15} | Amount: Rs.{e['amount']:,.2f}")
    print("--------------------------------------------------\n")

def handle_delete(args: argparse.Namespace) -> None:
    expenses = load_expenses()
    filtered = [e for e in expenses if e["id"] != args.id]
    if len(filtered) < len(expenses):
        save_expenses(filtered)
        logging.info("Deleted expense ID %d", args.id)
        print(f"[SUCCESS] Deleted Expense ID {args.id}")
    else:
        logging.warning("Delete failed: Expense ID %d not found", args.id)
        print(f"[WARNING] Expense ID {args.id} not found.")

def handle_summary(args: argparse.Namespace) -> None:
    expenses = load_expenses()
    if not expenses:
        print("No expenses recorded yet.")
        return
    summary = {}
    total = 0.0
    for e in expenses:
        cat = e["category"]
        amt = e["amount"]
        summary[cat] = summary.get(cat, 0.0) + amt
        total += amt

    print("\n------ EXPENSE SUMMARY ------")
    for cat, amt in summary.items():
        print(f"{cat:<15} Rs.{amt:,.2f}")
    print("-----------------------------")
    print(f"Total           Rs.{total:,.2f}\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Expense Tracker Manager")
    
    # What is used : add_subparsers
    # Why it is used: Creates subcommand routing for add, list, delete, summary commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Add Command
    add_parser = subparsers.add_parser("add", help="Add a new expense")
    add_parser.add_argument("--category", type=str, required=True, help="Expense category")
    add_parser.add_argument("--amount", type=float, required=True, help="Expense amount")

    # List Command
    list_parser = subparsers.add_parser("list", help="List all expenses")

    # Delete Command
    delete_parser = subparsers.add_parser("delete", help="Delete expense by ID")
    delete_parser.add_argument("--id", type=int, required=True, help="Expense ID to delete")

    # Summary Command
    summary_parser = subparsers.add_parser("summary", help="Display summary report")

    return parser

def main() -> None:
    print("=== MINI PROJECT: CLI EXPENSE TRACKER ===")
    parser = create_parser()
    
    # Direct execution simulation
    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        # 1. Add Food
        handle_add(parser.parse_args(["add", "--category", "Food", "--amount", "250"]))
        # 2. Add Travel
        handle_add(parser.parse_args(["add", "--category", "Travel", "--amount", "1200"]))
        # 3. List
        handle_list(parser.parse_args(["list"]))
        # 4. Summary
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

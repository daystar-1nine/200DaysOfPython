# ==============================================================================
# Program    : PyFinance CLI Command Handlers & Formatters
# Objective  : Parse CLI arguments and render formatted ASCII tables and reports.
# Concept    : Presentation Layer Separation
# Why Used   : Separates user input parsing and terminal formatting from business services.
# ==============================================================================

import argparse
import csv
import json
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.exceptions import PyFinanceError
from pyfinance.services.expense_service import ExpenseService
from pyfinance.services.report_service import ReportService
from pyfinance.services.currency_service import CurrencyService
from pyfinance.services.budget_service import BudgetService

def setup_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyFinance — Personal Finance Manager CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="PyFinance Subcommands")

    # add
    a_p = subparsers.add_parser("add", help="Add new expense")
    a_p.add_argument("--amount", type=float, required=True, help="Expense amount")
    a_p.add_argument("--category", type=str, required=True, help="Expense category")
    a_p.add_argument("--description", type=str, required=True, help="Expense description")
    a_p.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")

    # list
    subparsers.add_parser("list", help="List all expenses")

    # search
    s_p = subparsers.add_parser("search", help="Search expenses")
    s_p.add_argument("--category", type=str, default=None, help="Filter by category")
    s_p.add_argument("--from", dest="start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    s_p.add_argument("--to", dest="end_date", type=str, default=None, help="End date (YYYY-MM-DD)")
    s_p.add_argument("--keyword", type=str, default=None, help="Search keyword")

    # update
    u_p = subparsers.add_parser("update", help="Update existing expense")
    u_p.add_argument("id", type=int, help="Expense ID to update")
    u_p.add_argument("--amount", type=float, default=None, help="New amount")
    u_p.add_argument("--category", type=str, default=None, help="New category")
    u_p.add_argument("--description", type=str, default=None, help="New description")
    u_p.add_argument("--date", type=str, default=None, help="New date")

    # delete
    d_p = subparsers.add_parser("delete", help="Delete expense by ID")
    d_p.add_argument("id", type=int, help="Expense ID to delete")

    # report
    rp_p = subparsers.add_parser("report", help="Generate analytics reports")
    rp_p.add_argument("type", choices=["total", "category", "monthly"], help="Report type")

    # currency
    c_p = subparsers.add_parser("currency", help="Fetch live currency conversion")
    c_p.add_argument("base", type=str, help="Base currency code (e.g. USD)")
    c_p.add_argument("target", type=str, help="Target currency code (e.g. INR)")
    c_p.add_argument("--amount", type=float, default=1.0, help="Amount to convert")

    # budget
    b_p = subparsers.add_parser("budget", help="Manage category budgets")
    b_sub = b_p.add_subparsers(dest="budget_cmd", required=True)
    b_set = b_sub.add_parser("set", help="Set monthly budget limit")
    b_set.add_argument("--category", type=str, required=True, help="Category name")
    b_set.add_argument("--limit", type=float, required=True, help="Monthly limit")
    b_sub.add_parser("status", help="Check budget status")

    # export
    ex_p = subparsers.add_parser("export", help="Export expenses data")
    ex_p.add_argument("--format", choices=["csv", "json"], default="csv", help="Export format")
    ex_p.add_argument("--output", type=str, default="expenses.csv", help="Output file path")

    # import
    im_p = subparsers.add_parser("import", help="Import expenses data")
    im_p.add_argument("--format", choices=["csv", "json"], default="csv", help="Import format")
    im_p.add_argument("--input", type=str, required=True, help="Input file path")

    return parser

def print_banner() -> None:
    print("\n==================================================")
    print("                 PYFINANCE CLI                    ")
    print("==================================================")

def execute_cli_command(args: argparse.Namespace, expense_service: ExpenseService, report_service: ReportService, currency_service: CurrencyService, budget_service: BudgetService) -> None:
    try:
        if args.command == "add":
            e = expense_service.add_expense(args.amount, args.category, args.description, args.date)
            print(f"\n[SUCCESS] Expense #{e.id} added successfully: {e.category} -> Rs.{e.amount:,.2f}")
        
        elif args.command == "list":
            expenses = expense_service.list_expenses()
            print_banner()
            if not expenses:
                print("No expenses found in database.")
                return
            print(f"{'ID':<4} {'DATE':<12} {'CATEGORY':<14} {'AMOUNT':<10} {'DESCRIPTION'}")
            print("-" * 65)
            total = 0.0
            for e in expenses:
                print(f"{e.id:<4} {e.date:<12} {e.category:<14} Rs.{e.amount:<10.2f} {e.description}")
                total += e.amount
            print("-" * 65)
            print(f"Total Spent: Rs.{total:,.2f}\n")

        elif args.command == "search":
            results = expense_service.search_expenses(args.category, args.start_date, args.end_date, args.keyword)
            print_banner()
            print(f"--- SEARCH RESULTS ({len(results)} matches) ---")
            for e in results:
                print(f"ID #{e.id} | {e.date} | {e.category:<12} | Rs.{e.amount:,.2f} | {e.description}")
            print("-" * 50 + "\n")

        elif args.command == "update":
            e = expense_service.update_expense(args.id, args.amount, args.category, args.description, args.date)
            print(f"\n[SUCCESS] Updated Expense #{e.id}: {e.category} -> Rs.{e.amount:,.2f}")

        elif args.command == "delete":
            expense_service.delete_expense(args.id)
            print(f"\n[SUCCESS] Deleted Expense #{args.id}")

        elif args.command == "report":
            print_banner()
            if args.type == "total":
                tot = report_service.get_total_spending()
                print(f"Total Overall Spending: Rs.{tot:,.2f}\n")
            elif args.type == "category":
                cats = report_service.get_category_report()
                print("--- CATEGORY BREAKDOWN ---")
                for cat, amt in cats.items():
                    print(f"{cat:<18} Rs.{amt:,.2f}")
                print("-" * 35 + "\n")
            elif args.type == "monthly":
                months = report_service.get_monthly_report()
                print("--- MONTHLY SPENDING TRENDS ---")
                for m, amt in months.items():
                    print(f"{m:<12} Rs.{amt:,.2f}")
                print("-" * 35 + "\n")

        elif args.command == "currency":
            print_banner()
            rate_obj = currency_service.get_exchange_rate(args.base, args.target)
            converted = args.amount * rate_obj.rate
            print(f"{args.amount:,.2f} {rate_obj.base} = {converted:,.2f} {rate_obj.target} (Rate: 1 {rate_obj.base} = {rate_obj.rate:.4f} {rate_obj.target})\n")

        elif args.budget_cmd == "set":
            budget_service.set_budget(args.category, args.limit)
            print(f"\n[SUCCESS] Set monthly budget limit for '{args.category}' to Rs.{args.limit:,.2f}")
        elif args.budget_cmd == "status":
            statuses = budget_service.get_budget_statuses()
            print_banner()
            print("--- CATEGORY BUDGET STATUS ---")
            for b in statuses:
                flag = "EXCEEDED!" if b["is_exceeded"] else "OK"
                print(f"{b['category']:<14} | Limit: Rs.{b['limit']:<10.2f} | Spent: Rs.{b['spent']:<10.2f} | Rem: Rs.{b['remaining']:<10.2f} [{flag}]")
            print("-" * 65 + "\n")

        elif args.command == "export":
            expenses = expense_service.list_expenses()
            if args.format == "csv":
                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "amount", "category", "description", "date"])
                    for e in expenses:
                        writer.writerow([e.id, e.amount, e.category, e.description, e.date])
            else: # JSON
                data = [{"id": e.id, "amount": e.amount, "category": e.category, "description": e.description, "date": e.date} for e in expenses]
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            print(f"\n[SUCCESS] Exported {len(expenses)} expenses to {args.output}")

        elif args.command == "import":
            count = 0
            if args.format == "csv":
                with open(args.input, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        expense_service.add_expense(float(row["amount"]), row["category"], row["description"], row.get("date"))
                        count += 1
            else: # JSON
                with open(args.input, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for row in data:
                        expense_service.add_expense(float(row["amount"]), row["category"], row["description"], row.get("date"))
                        count += 1
            print(f"\n[SUCCESS] Imported {count} expenses from {args.input}")

    except PyFinanceError as err:
        print(f"\nERROR: {err}\n")

# ==============================================================================
# Program    : CLI Command Handlers (Presentation Layer)
# Objective  : Parse terminal user subcommands and delegate execution to services.
# Concept    : Separation of Presentation Concerns
# Why Used   : Keeps CLI formatting separate from business logic services.
# ==============================================================================

import argparse
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.services.expense_service import ExpenseService
from expense_tracker.services.report_service import ReportService

def setup_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional Layered Expense Tracker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # add-user
    au = subparsers.add_parser("add-user", help="Add new user")
    au.add_argument("--name", type=str, required=True, help="User name")
    au.add_argument("--email", type=str, required=True, help="User email")

    # add-category
    ac = subparsers.add_parser("add-category", help="Add new expense category")
    ac.add_argument("--name", type=str, required=True, help="Category name")

    # add
    a_p = subparsers.add_parser("add", help="Add new expense")
    a_p.add_argument("--user", type=int, default=1, help="User ID")
    a_p.add_argument("--category", type=str, required=True, help="Category name or ID")
    a_p.add_argument("--amount", type=float, required=True, help="Expense amount")
    a_p.add_argument("--description", type=str, default="", help="Description")

    # list
    l_p = subparsers.add_parser("list", help="List user expenses")
    l_p.add_argument("--user", type=int, default=1, help="User ID")

    # summary
    s_p = subparsers.add_parser("summary", help="Show user expense summary")
    s_p.add_argument("--user", type=int, default=1, help="User ID")

    # report
    r_p = subparsers.add_parser("report", help="Generate monthly report")
    r_p.add_argument("--user", type=int, default=1, help="User ID")
    r_p.add_argument("--month", type=str, required=True, help="Month (e.g. 08)")
    r_p.add_argument("--year", type=str, required=True, help="Year (e.g. 2026)")

    return parser

def execute_command(args: argparse.Namespace, expense_service: ExpenseService, report_service: ReportService) -> None:
    if args.command == "add-user":
        uid = expense_service.add_user(args.name, args.email)
        print(f"[SUCCESS] Created User #{uid}: {args.name}")
    elif args.command == "add-category":
        cid = expense_service.add_category(args.name)
        print(f"[SUCCESS] Created Category #{cid}: {args.name}")
    elif args.command == "add":
        cat_id = 1
        if args.category.isdigit():
            cat_id = int(args.category)
        eid = expense_service.add_expense(args.user, cat_id, args.amount, args.description)
        print(f"[SUCCESS] Added Expense #{eid}: Rs.{args.amount:,.2f}")
    elif args.command == "list":
        records = expense_service.get_user_expenses(args.user)
        print(f"\n--- EXPENSE RECORDS (USER #{args.user}) ---")
        for r in records:
            desc = f" ({r[4]})" if r[4] else ""
            print(f"ID: {r[0]:<3} | Cat: {r[2]:<12} | Amount: Rs.{r[3]:<8.2f} | Date: {r[5]}{desc}")
        print("-------------------------------------------\n")
    elif args.command == "summary":
        res = report_service.get_user_summary(args.user)
        print("\n+------------------------------------+")
        print("|          EXPENSE SUMMARY           |")
        print("+------------------------------------+")
        print(f"User: {res.get('user_name', 'N/A')}\n")
        for cat, amt in res.get("categories", []):
            print(f"{cat:<18} Rs.{amt:,.2f}")
        print("--------------------------------------")
        print(f"Total              Rs.{res.get('total', 0.0):,.2f}")
        print("+------------------------------------+\n")
    elif args.command == "report":
        res = report_service.get_monthly_report(args.user, args.month, args.year)
        print(f"\n------ {res.get('month')}/{res.get('year')} REPORT FOR {res.get('user_name')} ------")
        for cat, amt in res.get("categories", []):
            print(f"{cat:<18} Rs.{amt:,.2f}")
        print("--------------------------------------")
        print(f"Total Spent : Rs.{res.get('total', 0.0):,.2f}")
        print(f"Average     : Rs.{res.get('avg', 0.0):,.2f}")
        print(f"Transactions: {res.get('count', 0)}")
        print("--------------------------------------\n")

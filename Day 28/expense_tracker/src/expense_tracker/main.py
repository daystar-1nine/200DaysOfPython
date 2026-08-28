# ==============================================================================
# Program    : Application Bootstrap (Dependency Injection Wiring)
# Objective  : Instantiate Database, inject dependencies into Services, and execute CLI.
# Concept    : Composition Root & Dependency Injection
# Why Used   : Wires Database layer -> Services -> CLI Presentation layer.
# ==============================================================================

import os
import sys

# Ensure package directory is in sys.path when running main.py directly
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.config import DATABASE_PATH
from expense_tracker.database import Database
from expense_tracker.services.expense_service import ExpenseService
from expense_tracker.services.report_service import ReportService
from expense_tracker.cli.commands import setup_cli_parser, execute_command

def main() -> None:
    print("=== REFACTORED EXPENSE TRACKER (DAY 28 ARCHITECTURE) ===")
    
    # What is used : Dependency Injection Wiring
    # Why it is used: Database is instantiated once and injected into services
    database = Database(DATABASE_PATH)
    expense_service = ExpenseService(database)
    report_service = ReportService(database)

    parser = setup_cli_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        # Ensure default user and category exist
        try:
            expense_service.add_user("Suraj Sawant", "suraj@example.com")
        except Exception:
            pass
        try:
            expense_service.add_category("Food")
            expense_service.add_category("Travel")
        except Exception:
            pass

        # Simulate commands
        execute_command(parser.parse_args(["add", "--category", "Food", "--amount", "250.0", "--description", "Lunch"]), expense_service, report_service)
        execute_command(parser.parse_args(["list"]), expense_service, report_service)
        execute_command(parser.parse_args(["summary"]), expense_service, report_service)
        execute_command(parser.parse_args(["report", "--month", "08", "--year", "2026"]), expense_service, report_service)
    else:
        args = parser.parse_args()
        execute_command(args, expense_service, report_service)

if __name__ == "__main__":
    main()

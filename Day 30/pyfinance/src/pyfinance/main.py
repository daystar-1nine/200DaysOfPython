# ==============================================================================
# Program    : PyFinance Application Main Bootstrap
# Objective  : Instantiate Database, Repository, Services, and execute CLI command.
# Concept    : Composition Root & Dependency Injection
# Why Used   : Connects Database -> Repository -> Services -> CLI Presentation layer.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.config import DATABASE_PATH
from pyfinance.database import Database
from pyfinance.repositories.expense_repository import ExpenseRepository
from pyfinance.services.expense_service import ExpenseService
from pyfinance.services.report_service import ReportService
from pyfinance.services.currency_service import CurrencyService
from pyfinance.services.budget_service import BudgetService
from pyfinance.cli.commands import setup_cli_parser, execute_cli_command

def main() -> None:
    # What is used : Composition Root Dependency Injection Wiring
    # Why it is used: Constructs singletons and injects dependencies down the application stack
    database = Database(DATABASE_PATH)
    repository = ExpenseRepository(database)
    expense_service = ExpenseService(repository)
    report_service = ReportService(repository)
    currency_service = CurrencyService()
    budget_service = BudgetService(repository)

    parser = setup_cli_parser()

    if len(sys.argv) == 1:
        print("=== PYFINANCE CLI SIMULATION RUN ===")
        # Seed initial sample data if empty
        if len(expense_service.list_expenses()) == 0:
            expense_service.add_expense(250.0, "Food", "Lunch at Cafe")
            expense_service.add_expense(500.0, "Travel", "Bus Ticket")
            expense_service.add_expense(1200.0, "Shopping", "Shoes")
            budget_service.set_budget("Food", 5000.0)

        execute_cli_command(parser.parse_args(["list"]), expense_service, report_service, currency_service, budget_service)
        execute_cli_command(parser.parse_args(["report", "category"]), expense_service, report_service, currency_service, budget_service)
        execute_cli_command(parser.parse_args(["budget", "status"]), expense_service, report_service, currency_service, budget_service)
        execute_cli_command(parser.parse_args(["currency", "USD", "INR", "--amount", "100"]), expense_service, report_service, currency_service, budget_service)
    else:
        args = parser.parse_args()
        execute_cli_command(args, expense_service, report_service, currency_service, budget_service)

if __name__ == "__main__":
    main()

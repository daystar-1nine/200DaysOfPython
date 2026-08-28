# ==============================================================================
# Test Suite : Unit Tests for Report Service Layer
# Objective  : Test Total Spending, Category Breakdown, and Monthly Trends.
# Concept    : Analytics & Financial Reporting Testing
# Why Used   : Asserts report computations from expense records.
# ==============================================================================

import os
import sys
import unittest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 30", "pyfinance", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.database import Database
from pyfinance.repositories.expense_repository import ExpenseRepository
from pyfinance.services.expense_service import ExpenseService
from pyfinance.services.report_service import ReportService

def test_report_service_calculations(expense_service, report_service):
    expense_service.add_expense(250.0, "Food", "Lunch", "2026-08-01")
    expense_service.add_expense(500.0, "Travel", "Bus", "2026-08-02")
    expense_service.add_expense(150.0, "Food", "Dinner", "2026-08-03")

    total = report_service.get_total_spending()
    assert total == 900.0

    cats = report_service.get_category_report()
    assert cats["Food"] == 400.0
    assert cats["Travel"] == 500.0

    months = report_service.get_monthly_report()
    assert months["2026-08"] == 900.0

class TestReportServiceUnitRunner(unittest.TestCase):
    def test_report_service_standalone(self):
        db_file = "temp_pyfinance_rep_unit.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        repo = ExpenseRepository(db)
        exp_service = ExpenseService(repo)
        rep_service = ReportService(repo)
        try:
            exp_service.add_expense(300.0, "Bills", "Electricity", "2026-08-10")
            self.assertEqual(rep_service.get_total_spending(), 300.0)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

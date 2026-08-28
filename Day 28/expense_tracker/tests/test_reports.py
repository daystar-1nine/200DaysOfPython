# ==============================================================================
# Test Suite : Unit Tests for Report Service Layer
# Objective  : Verify summary and monthly reporting calculations.
# Concept    : Unit Testing Aggregations
# Why Used   : Validates statistical computations and category totals.
# ==============================================================================

import os
import sys
import unittest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database
from expense_tracker.services.expense_service import ExpenseService
from expense_tracker.services.report_service import ReportService

class TestReportService(unittest.TestCase):
    def setUp(self):
        self.test_db_file = f"test_reports_db_{self._testMethodName}.db"
        if os.path.exists(self.test_db_file):
            try:
                os.remove(self.test_db_file)
            except OSError:
                pass
        self.db = Database(self.test_db_file)
        self.expense_service = ExpenseService(self.db)
        self.report_service = ReportService(self.db)

    def tearDown(self):
        if os.path.exists(self.test_db_file):
            try:
                os.remove(self.test_db_file)
            except OSError:
                pass

    def test_summary_report(self):
        uid = self.expense_service.add_user("Report User", "report_unique_1@example.com")
        cid = self.expense_service.add_category("Travel")
        self.expense_service.add_expense(uid, cid, 500.0)
        self.expense_service.add_expense(uid, cid, 300.0)

        res = self.report_service.get_user_summary(uid)
        self.assertEqual(res["user_name"], "Report User")
        self.assertEqual(res["total"], 800.0)

if __name__ == "__main__":
    unittest.main()

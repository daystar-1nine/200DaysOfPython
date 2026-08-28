# ==============================================================================
# Test Suite : Unit Tests for Report Service Layer
# Objective  : Test user total expenses, category totals, and monthly report analytics.
# Concept    : Unit Testing Analytical Aggregations
# Why Used   : Asserts report computation logic.
# ==============================================================================

import os
import sys
import unittest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 28", "expense_tracker", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database
from expense_tracker.services.expense_service import ExpenseService
from expense_tracker.services.report_service import ReportService

def test_user_summary_report(seeded_services):
    exp_service = seeded_services["expense_service"]
    rep_service = seeded_services["report_service"]
    uid = seeded_services["user_id"]
    cid = seeded_services["category_id"]

    exp_service.add_expense(uid, cid, 500.0, "Fuel")
    exp_service.add_expense(uid, cid, 300.0, "Service")

    summary = rep_service.get_user_summary(uid)
    assert summary["user_name"] == "Suraj Sawant"
    assert summary["total"] == 800.0

class TestReportServiceRunner(unittest.TestCase):
    def test_report_service_standalone(self):
        db_file = "temp_ut_reports.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        exp_service = ExpenseService(db)
        rep_service = ReportService(db)
        try:
            uid = exp_service.add_user("Report UT", "rep_ut@example.com")
            cid = exp_service.add_category("Travel")
            exp_service.add_expense(uid, cid, 400.0)
            summary = rep_service.get_user_summary(uid)
            self.assertEqual(summary["total"], 400.0)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

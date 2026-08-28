# ==============================================================================
# Test Suite : CLI Integration Pytest Suite
# Objective  : Execute end-to-end CLI subcommands and capture output formatting.
# Concept    : CLI Integration Testing
# Why Used   : Validates presentation layer subcommands (add-user, add-category, add, list, summary, report).
# ==============================================================================

import io
import os
import sys
import unittest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 28", "expense_tracker", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.cli.commands import setup_cli_parser, execute_command

def test_cli_end_to_end(seeded_services):
    parser = setup_cli_parser()
    exp_service = seeded_services["expense_service"]
    rep_service = seeded_services["report_service"]
    uid = seeded_services["user_id"]
    cid = seeded_services["category_id"]

    captured = io.StringIO()
    sys.stdout = captured

    try:
        execute_command(parser.parse_args(["add", "--user", str(uid), "--category", str(cid), "--amount", "350.0", "--description", "CLI Test Dinner"]), exp_service, rep_service)
        execute_command(parser.parse_args(["list", "--user", str(uid)]), exp_service, rep_service)
        execute_command(parser.parse_args(["summary", "--user", str(uid)]), exp_service, rep_service)
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "[SUCCESS] Added Expense" in output
    assert "CLI Test Dinner" in output
    assert "EXPENSE SUMMARY" in output

class TestCLIIntegrationRunner(unittest.TestCase):
    def test_cli_standalone(self):
        from expense_tracker.database import Database
        from expense_tracker.services.expense_service import ExpenseService
        from expense_tracker.services.report_service import ReportService

        db_file = "temp_cli_test.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        exp_service = ExpenseService(db)
        rep_service = ReportService(db)
        try:
            uid = exp_service.add_user("CLI User", "cli@example.com")
            cid = exp_service.add_category("Food")
            parser = setup_cli_parser()

            captured = io.StringIO()
            sys.stdout = captured
            try:
                execute_command(parser.parse_args(["add", "--user", str(uid), "--category", str(cid), "--amount", "100.0"]), exp_service, rep_service)
            finally:
                sys.stdout = sys.__stdout__

            self.assertIn("[SUCCESS] Added Expense", captured.getvalue())
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

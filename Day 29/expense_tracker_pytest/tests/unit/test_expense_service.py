# ==============================================================================
# Test Suite : Unit Tests for Expense Service Layer
# Objective  : Test valid expense creation, negative amount handling, missing category, invalid user.
# Concept    : Unit Testing Services with Pytest Fixtures
# Why Used   : Asserts business rule enforcement in ExpenseService.
# ==============================================================================

import os
import sys
import unittest
import pytest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 28", "expense_tracker", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database
from expense_tracker.services.expense_service import ExpenseService

def test_add_valid_expense(seeded_services):
    service = seeded_services["expense_service"]
    uid = seeded_services["user_id"]
    cid = seeded_services["category_id"]

    eid = service.add_expense(uid, cid, 250.0, "Supermarket Lunch")
    assert eid > 0

    records = service.get_user_expenses(uid)
    assert len(records) == 1
    assert records[0][3] == 250.0

def test_negative_amount_expense_raises_error(seeded_services):
    service = seeded_services["expense_service"]
    with pytest.raises(ValueError, match="Amount must be greater than zero"):
        service.add_expense(1, 1, -100.0)

def test_missing_user_name_raises_error(expense_service):
    with pytest.raises(ValueError, match="Name and email are required"):
        expense_service.add_user("", "test@example.com")

class TestExpenseServiceRunner(unittest.TestCase):
    def test_expense_service_standalone(self):
        db_file = "temp_ut_expenses.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        service = ExpenseService(db)
        try:
            uid = service.add_user("UT User", "ut@example.com")
            cid = service.add_category("Food")
            eid = service.add_expense(uid, cid, 200.0, "Lunch")
            self.assertGreater(eid, 0)
            with self.assertRaises(ValueError):
                service.add_expense(uid, cid, -50.0)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

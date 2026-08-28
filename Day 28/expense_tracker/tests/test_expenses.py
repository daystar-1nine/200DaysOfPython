# ==============================================================================
# Test Suite : Unit Tests for Expense Service Layer
# Objective  : Verify user, category, and expense operations using isolated test database.
# Concept    : Unit Testing with In-Memory SQLite Database
# Why Used   : Tests business logic via Dependency Injection of temporary test database.
# ==============================================================================

import os
import sys
import unittest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database
from expense_tracker.services.expense_service import ExpenseService

class TestExpenseService(unittest.TestCase):
    def setUp(self):
        self.test_db_file = f"test_expenses_db_{self._testMethodName}.db"
        if os.path.exists(self.test_db_file):
            try:
                os.remove(self.test_db_file)
            except OSError:
                pass
        self.db = Database(self.test_db_file)
        self.service = ExpenseService(self.db)

    def tearDown(self):
        if os.path.exists(self.test_db_file):
            try:
                os.remove(self.test_db_file)
            except OSError:
                pass

    def test_add_user_and_expense(self):
        uid = self.service.add_user("Test User", "test_unique_1@example.com")
        cid = self.service.add_category("Food")
        eid = self.service.add_expense(uid, cid, 150.0, "Snacks")
        
        self.assertGreater(uid, 0)
        self.assertGreater(cid, 0)
        self.assertGreater(eid, 0)

        records = self.service.get_user_expenses(uid)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][3], 150.0)

    def test_negative_amount_raises_error(self):
        with self.assertRaises(ValueError):
            self.service.add_expense(1, 1, -50.0)

if __name__ == "__main__":
    unittest.main()

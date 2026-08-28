# ==============================================================================
# Test Suite : Unit Tests for Input Validation
# Objective  : Test input bounds, empty categories, and invalid parameters.
# Concept    : Input Validation Boundary Testing
# Why Used   : Ensures application fails safely on invalid input data.
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

def test_empty_category_name_validation(expense_service):
    with pytest.raises(ValueError, match="Category name is required"):
        expense_service.add_category("")

def test_zero_amount_validation(expense_service):
    with pytest.raises(ValueError, match="Amount must be greater than zero"):
        expense_service.add_expense(1, 1, 0.0)

class TestValidationRunner(unittest.TestCase):
    def test_validation_standalone(self):
        db_file = "temp_ut_valid.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        service = ExpenseService(db)
        try:
            with self.assertRaises(ValueError):
                service.add_category("")
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

# ==============================================================================
# Test Suite : Unit Tests for PyFinance Input Validation
# Objective  : Test empty descriptions, zero amounts, and invalid parameters.
# Concept    : Input Boundary Testing
# Why Used   : Ensures application raises ValidationError safely.
# ==============================================================================

import os
import sys
import unittest
import pytest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 30", "pyfinance", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.database import Database
from pyfinance.repositories.expense_repository import ExpenseRepository
from pyfinance.services.expense_service import ExpenseService
from pyfinance.exceptions import ValidationError

def test_empty_description_raises_error(expense_service):
    with pytest.raises(ValidationError, match="Description cannot be empty"):
        expense_service.add_expense(100.0, "Food", "")

def test_zero_amount_raises_error(expense_service):
    with pytest.raises(ValidationError, match="Amount must be greater than zero"):
        expense_service.add_expense(0.0, "Food", "Snacks")

class TestValidationUnitRunner(unittest.TestCase):
    def test_validation_standalone(self):
        db_file = "temp_pyfinance_val_unit.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        repo = ExpenseRepository(db)
        service = ExpenseService(repo)
        try:
            with self.assertRaises(ValidationError):
                service.add_expense(0.0, "Food", "Snacks")
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

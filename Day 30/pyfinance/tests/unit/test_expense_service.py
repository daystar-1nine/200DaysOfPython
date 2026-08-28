# ==============================================================================
# Test Suite : Unit Tests for Expense Service Layer
# Objective  : Test expense creation, validation, updating, and deletion.
# Concept    : Unit Testing Business Logic Services
# Why Used   : Asserts business rule enforcement in ExpenseService.
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
from pyfinance.exceptions import ValidationError, NotFoundError

def test_add_expense_valid(expense_service):
    exp = expense_service.add_expense(250.0, "Food", "Lunch at Cafe")
    assert exp.id is not None
    assert exp.amount == 250.0
    assert exp.category == "Food"

def test_add_expense_negative_amount(expense_service):
    with pytest.raises(ValidationError, match="Amount must be greater than zero"):
        expense_service.add_expense(-50.0, "Food", "Lunch")

def test_add_expense_empty_category(expense_service):
    with pytest.raises(ValidationError, match="Category cannot be empty"):
        expense_service.add_expense(100.0, "", "Lunch")

def test_delete_non_existing_expense_raises_not_found(expense_service):
    with pytest.raises(NotFoundError):
        expense_service.delete_expense(9999)

class TestExpenseServiceUnitRunner(unittest.TestCase):
    def test_expense_service_standalone(self):
        db_file = "temp_pyfinance_unit.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        repo = ExpenseRepository(db)
        service = ExpenseService(repo)
        try:
            exp = service.add_expense(100.0, "Travel", "Bus Ticket")
            self.assertIsNotNone(exp.id)
            with self.assertRaises(ValidationError):
                service.add_expense(-10.0, "Travel", "Bus")
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

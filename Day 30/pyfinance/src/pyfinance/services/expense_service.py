# ==============================================================================
# Program    : Expense Service (Business Logic Layer)
# Objective  : Business validation and orchestration for expense CRUD operations.
# Concept    : Service Layer with Dependency Injection
# Why Used   : Encapsulates validation and delegates storage to ExpenseRepository.
# ==============================================================================

from datetime import datetime
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.exceptions import ValidationError, NotFoundError
from pyfinance.models.expense import Expense
from pyfinance.repositories.expense_repository import ExpenseRepository
from pyfinance.logger import get_logger

logger = get_logger("ExpenseService")

class ExpenseService:
    def __init__(self, repository: ExpenseRepository):
        # What is used : Dependency Injection (self.repository = repository)
        # Why it is used: Injects database repository dependency allowing mock repository replacement in unit tests
        self.repository = repository

    def add_expense(self, amount: float, category: str, description: str, date: str | None = None) -> Expense:
        if amount <= 0:
            raise ValidationError("Amount must be greater than zero.")
        if not category or not category.strip():
            raise ValidationError("Category cannot be empty.")
        if not description or not description.strip():
            raise ValidationError("Description cannot be empty.")

        date_str = date if date else datetime.now().strftime("%Y-%m-%d")
        expense = Expense(id=None, amount=amount, category=category.strip(), description=description.strip(), date=date_str)
        eid = self.repository.add(expense)
        expense.id = eid
        logger.info("Created Expense ID #%d: %s -> Rs.%.2f", eid, category, amount)
        return expense

    def list_expenses(self) -> list[Expense]:
        return self.repository.get_all_records()

    def get_expense(self, expense_id: int) -> Expense:
        expense = self.repository.get_by_id(expense_id)
        if not expense:
            raise NotFoundError(f"Expense with ID #{expense_id} was not found.")
        return expense

    def update_expense(self, expense_id: int, amount: float | None = None, category: str | None = None, description: str | None = None, date: str | None = None) -> Expense:
        if amount is not None and amount <= 0:
            raise ValidationError("Amount must be greater than zero.")

        success = self.repository.update(expense_id, amount, category, description, date)
        if not success:
            raise NotFoundError(f"Expense with ID #{expense_id} was not found.")
        
        logger.info("Updated Expense ID #%d", expense_id)
        return self.get_expense(expense_id)

    def delete_expense(self, expense_id: int) -> None:
        success = self.repository.delete(expense_id)
        if not success:
            raise NotFoundError(f"Expense with ID #{expense_id} was not found.")
        logger.info("Deleted Expense ID #%d", expense_id)

    def search_expenses(self, category: str | None = None, start_date: str | None = None, end_date: str | None = None, keyword: str | None = None) -> list[Expense]:
        return self.repository.search(category, start_date, end_date, keyword)

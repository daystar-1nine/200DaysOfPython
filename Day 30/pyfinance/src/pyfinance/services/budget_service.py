# ==============================================================================
# Program    : Budget Service Layer (Bonus Feature)
# Objective  : Business logic for monthly category budgets and spending limits.
# Concept    : Financial Planning Logic
# Why Used   : Tracks category spending vs budget thresholds.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.exceptions import ValidationError
from pyfinance.repositories.expense_repository import ExpenseRepository

class BudgetService:
    def __init__(self, repository: ExpenseRepository):
        self.repository = repository

    def set_budget(self, category: str, limit: float) -> None:
        if limit <= 0:
            raise ValidationError("Budget limit must be greater than zero.")
        if not category or not category.strip():
            raise ValidationError("Category name is required.")
        self.repository.set_budget(category.strip(), limit)

    def get_budget_statuses(self) -> list[dict]:
        budgets = self.repository.get_budgets()
        expenses = self.repository.get_all_records()
        
        # Calculate spending per category
        spent_map: dict[str, float] = {}
        for e in expenses:
            spent_map[e.category] = spent_map.get(e.category, 0.0) + e.amount

        statuses = []
        for b in budgets:
            spent = spent_map.get(b.category, 0.0)
            remaining = b.monthly_limit - spent
            statuses.append({
                "category": b.category,
                "limit": b.monthly_limit,
                "spent": spent,
                "remaining": remaining,
                "is_exceeded": spent > b.monthly_limit
            })
        return statuses

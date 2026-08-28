# ==============================================================================
# Program    : Report Service (Analytical Service Layer)
# Objective  : Compute Total Spending, Category Summaries, and Monthly Trend Reports.
# Concept    : Financial Analytics Aggregations
# Why Used   : Computes reporting statistics from ExpenseRepository records.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.repositories.expense_repository import ExpenseRepository

class ReportService:
    def __init__(self, repository: ExpenseRepository):
        self.repository = repository

    def get_total_spending(self) -> float:
        expenses = self.repository.get_all_records()
        return sum(e.amount for e in expenses)

    def get_category_report(self) -> dict[str, float]:
        expenses = self.repository.get_all_records()
        cat_totals: dict[str, float] = {}
        for e in expenses:
            cat_totals[e.category] = cat_totals.get(e.category, 0.0) + e.amount
        return dict(sorted(cat_totals.items(), key=lambda item: item[1], reverse=True))

    def get_monthly_report(self) -> dict[str, float]:
        expenses = self.repository.get_all_records()
        monthly_totals: dict[str, float] = {}
        for e in expenses:
            # e.date format: YYYY-MM-DD
            month_key = e.date[:7] if len(e.date) >= 7 else "Unknown"
            monthly_totals[month_key] = monthly_totals.get(month_key, 0.0) + e.amount
        return dict(sorted(monthly_totals.items()))

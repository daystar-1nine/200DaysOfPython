# ==============================================================================
# Program    : Report Service (Analytical Business Logic Layer)
# Objective  : Generate summary metrics and monthly expense reports.
# Concept    : Analytics & Service Abstraction
# Why Used   : Encapsulates complex summary calculations away from CLI commands.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database

class ReportService:
    def __init__(self, database: Database):
        self.database = database

    def get_user_summary(self, user_id: int) -> dict:
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            user_row = cursor.fetchone()
            if not user_row:
                return {}
            
            user_name = user_row[0]
            cursor.execute("""
                SELECT c.name, SUM(e.amount)
                FROM expenses e
                INNER JOIN categories c ON e.category_id = c.id
                WHERE e.user_id = ?
                GROUP BY c.id
                ORDER BY SUM(e.amount) DESC
            """, (user_id,))
            cat_totals = cursor.fetchall()

            cursor.execute("SELECT SUM(amount) FROM expenses WHERE user_id = ?", (user_id,))
            total_sum = cursor.fetchone()[0] or 0.0

            return {
                "user_name": user_name,
                "categories": cat_totals,
                "total": total_sum
            }

    def get_monthly_report(self, user_id: int, month: str, year: str) -> dict:
        month_str = f"{int(month):02d}"
        date_pattern = f"{year}-{month_str}%"

        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            u = cursor.fetchone()
            if not u:
                return {}

            user_name = u[0]
            cursor.execute("""
                SELECT c.name, SUM(e.amount)
                FROM expenses e
                INNER JOIN categories c ON e.category_id = c.id
                WHERE e.user_id = ? AND e.date LIKE ?
                GROUP BY c.id
                ORDER BY SUM(e.amount) DESC
            """, (user_id, date_pattern))
            cat_totals = cursor.fetchall()

            cursor.execute("SELECT SUM(amount), AVG(amount), MAX(amount), COUNT(*) FROM expenses WHERE user_id = ? AND date LIKE ?", (user_id, date_pattern))
            total_amt, avg_amt, max_amt, tx_count = cursor.fetchone()

            return {
                "user_name": user_name,
                "month": month_str,
                "year": year,
                "categories": cat_totals,
                "total": total_amt or 0.0,
                "avg": avg_amt or 0.0,
                "max": max_amt or 0.0,
                "count": tx_count or 0
            }

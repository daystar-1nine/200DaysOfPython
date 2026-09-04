"""
===============================================================================
DAY 53 — CODING CHALLENGE 3: REVENUE BY CATEGORY
===============================================================================
Topic: Data Grouping and Aggregation using defaultdict
Goal: Group sales records by product category and compute total revenue per category.
===============================================================================
"""

from collections import defaultdict
from typing import List, Dict, Any


def calculate_category_revenue(sales: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregated total revenue grouped by category."""
    # What is used: collections.defaultdict(float) for key initialization.
    # Why it is used: Automatically sets missing category total default to 0.0.
    # How it works: Loops through sales records, accumulating total amount into category key.
    category_totals: Dict[str, float] = defaultdict(float)

    for sale in sales:
        category = str(sale.get("category", "Uncategorized")).strip()
        total = float(sale.get("total", 0.0))
        category_totals[category] += total

    return dict(category_totals)


if __name__ == "__main__":
    sales_data = [
        {"category": "Electronics", "total": 1000},
        {"category": "Furniture", "total": 2000},
        {"category": "Electronics", "total": 3000},
    ]
    result = calculate_category_revenue(sales_data)
    print("Category Revenue Aggregations:")
    for cat, rev in result.items():
        print(f"  {cat}: ${rev:,.2f}")

    assert result["Electronics"] == 4000.0, "Electronics total mismatched"
    assert result["Furniture"] == 2000.0, "Furniture total mismatched"
    print("[OK] Challenge 3 Passed Successfully!")

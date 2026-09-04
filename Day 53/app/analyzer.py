"""
===============================================================================
DAY 53 — SALES ANALYZER MODULE
===============================================================================
This module computes overall revenue, average order value, highest order,
category breakdown, and product performance analytics.
===============================================================================
"""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
from app.models import Sale


def total_revenue(sales: List[Sale]) -> float:
    """Calculate aggregate total revenue across all sales."""
    # What is used: Generator expression with sum().
    # Why it is used: Efficiently computes total financial revenue sum.
    # How it works: Sums computed sale.total derived property across list.
    return sum(sale.total for sale in sales)


def average_order_value(sales: List[Sale]) -> float:
    """Calculate mean Average Order Value (AOV)."""
    # What is used: Arithmetic division with empty list safety check.
    # Why it is used: Prevents ZeroDivisionError when sales list is empty.
    # How it works: Returns total_revenue / len(sales) or 0.0 if empty.
    if not sales:
        return 0.0
    return total_revenue(sales) / len(sales)


def highest_value_order(sales: List[Sale]) -> Optional[Sale]:
    """Identify transaction record with the highest total amount."""
    # What is used: max() function with key=lambda s: s.total.
    # Why it is used: Finds maximum revenue order in $O(N)$ linear execution time.
    # How it works: Evaluates sale.total for each item and returns extremal Sale object.
    if not sales:
        return None
    return max(sales, key=lambda s: s.total)


def best_selling_product(sales: List[Sale]) -> Tuple[str, int]:
    """Find product name with highest aggregate units sold."""
    # What is used: collections.Counter.
    # Why it is used: Tallies product quantities and extracts top item.
    # How it works: Aggregates quantities by product string; returns top tuple or ("None", 0).
    if not sales:
        return ("None", 0)

    counts: Counter = Counter()
    for sale in sales:
        counts[sale.product] += sale.quantity

    top_item, top_qty = counts.most_common(1)[0]
    return (top_item, top_qty)


def category_revenue(sales: List[Sale]) -> Dict[str, float]:
    """Compute total revenue grouped by product category."""
    # What is used: collections.defaultdict(float).
    # Why it is used: Groups sales by category without manual key existence checks.
    # How it works: Loops through sales, adding sale.total to corresponding category key.
    totals: Dict[str, float] = defaultdict(float)
    for sale in sales:
        totals[sale.category] += sale.total
    return dict(totals)


def product_summary(sales: List[Sale]) -> Dict[str, Dict[str, Any]]:
    """Generate detailed product-wise units sold and revenue breakdown."""
    # What is used: Dictionary aggregation accumulator.
    # Why it is used: Summarizes both quantity sold and total revenue per product.
    # How it works: Returns nested dict {product_name: {"quantity": int, "revenue": float}}.
    summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"quantity": 0, "revenue": 0.0})

    for sale in sales:
        summary[sale.product]["quantity"] += sale.quantity
        summary[sale.product]["revenue"] += sale.total

    return {k: dict(v) for k, v in summary.items()}


def top_products_by_revenue(sales: List[Sale], top_n: int = 5) -> List[Tuple[str, float]]:
    """Retrieve top N products ordered descending by total revenue generated."""
    # What is used: sorted() function with key lambda and tuple ordering.
    # Why it is used: Ranks top selling products by gross monetary value.
    # How it works: Computes summary dict and sorts items descending by revenue value.
    summary = product_summary(sales)
    sorted_items = sorted(
        summary.items(),
        key=lambda item: item[1]["revenue"],
        reverse=True
    )
    return [(name, data["revenue"]) for name, data in sorted_items[:top_n]]

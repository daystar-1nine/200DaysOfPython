"""
===============================================================================
DAY 53 — TEST ANALYZER MODULE
===============================================================================
This test module verifies financial total revenue, average order value, highest
order identification, category breakdown, top products ranking, and empty edge cases.
===============================================================================
"""

from app.analyzer import (
    total_revenue,
    average_order_value,
    highest_value_order,
    best_selling_product,
    category_revenue,
    product_summary,
    top_products_by_revenue,
)


def test_total_revenue(sample_sales):
    """Verify total revenue aggregation across sales records."""
    # What is used: total_revenue function.
    # Why it is used: Validates sum of order total derived properties.
    # How it works: 55000 + 2400 + 2500 + 14000 + 15000 = 88900.0.
    assert total_revenue(sample_sales) == 88900.0
    assert total_revenue([]) == 0.0


def test_average_order_value(sample_sales):
    """Verify mean Average Order Value calculation."""
    # What is used: average_order_value function.
    # Why it is used: Validates AOV calculation and zero-division handling.
    # How it works: 88900.0 / 5 = 17780.0.
    assert average_order_value(sample_sales) == 17780.0
    assert average_order_value([]) == 0.0


def test_highest_value_order(sample_sales):
    """Verify identification of highest revenue order."""
    # What is used: highest_value_order function.
    # Why it is used: Extracts order object with maximum total value.
    # How it works: Order 1001 (55000.0) has highest value.
    highest = highest_value_order(sample_sales)
    assert highest is not None
    assert highest.order_id == 1001
    assert highest.total == 55000.0
    assert highest_value_order([]) is None


def test_best_selling_product(sample_sales):
    """Verify identification of best-selling product by quantity."""
    # What is used: best_selling_product function.
    # Why it is used: Extracts product with highest aggregate units sold.
    # How it works: Mouse (2 units) and Chair (2 units); Mouse or Chair top product.
    prod, qty = best_selling_product(sample_sales)
    assert qty == 2
    assert best_selling_product([]) == ("None", 0)


def test_category_revenue(sample_sales):
    """Verify revenue breakdown grouped by product category."""
    # What is used: category_revenue function.
    # Why it is used: Validates category aggregation totals.
    # How it works: Electronics total = 55000+2400+2500 = 59900.0; Furniture total = 14000+15000 = 29000.0.
    cats = category_revenue(sample_sales)
    assert cats["Electronics"] == 59900.0
    assert cats["Furniture"] == 29000.0


def test_product_summary_and_top_products(sample_sales):
    """Verify detailed product summary and top N products ranking."""
    # What is used: product_summary and top_products_by_revenue functions.
    # Why it is used: Validates ranking of products descending by revenue.
    # How it works: Top 1 product by revenue is Laptop ($55,000).
    summary = product_summary(sample_sales)
    assert summary["Laptop"]["quantity"] == 1
    assert summary["Laptop"]["revenue"] == 55000.0

    top_1 = top_products_by_revenue(sample_sales, top_n=1)
    assert len(top_1) == 1
    assert top_1[0] == ("Laptop", 55000.0)

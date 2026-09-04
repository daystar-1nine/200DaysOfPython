"""
===============================================================================
DAY 55 — CODING CHALLENGE 8: MINI 30-DAY SALES DATASET ANALYZER
===============================================================================
Topic: Realistic Data Analyst-Style 2D Matrix Analytics
Goal: Generate (30 days x 7 products) sales matrix and compute analytical metrics.
===============================================================================
"""

import numpy as np


def analyze_sales_dataset():
    """Generate 30-day x 7-product sales matrix and compute analytical insights."""
    # What is used: np.random.default_rng(42) and 2D axis aggregations.
    # Why it is used: Simulates real-world data analyst daily/product sales reporting.
    # How it works: Generates random integers (100 to 10000), evaluates sum across axes,
    #               finds best product/day via argmax, and ranks top 3 products via argsort.
    rng = np.random.default_rng(42)
    sales = rng.integers(100, 10000, size=(30, 7))

    products = np.array([f"Prod_{i}" for i in range(1, 8)])

    # 1. Total sales per product (axis=0 down rows across 30 days)
    product_totals = np.sum(sales, axis=0)

    # 2. Total sales per day (axis=1 across 7 products)
    daily_totals = np.sum(sales, axis=1)

    # 3. Best selling product
    best_product_idx = np.argmax(product_totals)
    best_product = products[best_product_idx]

    # 4. Best sales day (Day index 1-30)
    best_day_idx = np.argmax(daily_totals) + 1

    # 5. Average daily sales
    avg_daily_sales = np.mean(daily_totals)

    # 6. Top 3 products by revenue
    top_3_indices = np.argsort(product_totals)[::-1][:3]
    top_3_products = products[top_3_indices]

    return {
        "sales_shape": sales.shape,
        "product_totals": product_totals,
        "daily_totals": daily_totals,
        "best_product": best_product,
        "best_day": best_day_idx,
        "avg_daily_sales": avg_daily_sales,
        "top_3_products": list(top_3_products),
    }


if __name__ == "__main__":
    res = analyze_sales_dataset()
    print("30-Day Sales Dataset Analysis Results:")
    print("  Dataset Shape:      ", res["sales_shape"])
    print("  Best Selling Product:", res["best_product"])
    print("  Best Sales Day:      Day", res["best_day"])
    print(f"  Avg Daily Sales:     ${res['avg_daily_sales']:,.2f}")
    print("  Top 3 Products:      ", res["top_3_products"])

    assert res["sales_shape"] == (30, 7), "Shape failed"
    assert len(res["top_3_products"]) == 3, "Top 3 count failed"
    print("[OK] Challenge 8 Passed Successfully!")

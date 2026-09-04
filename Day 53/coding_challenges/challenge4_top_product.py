"""
===============================================================================
DAY 53 — CODING CHALLENGE 4: TOP SELLING PRODUCT
===============================================================================
Topic: Product Quantity Accumulation and Max Extraction
Goal: Aggregate total units sold per product and identify the highest selling product.
===============================================================================
"""

from collections import Counter
from typing import List, Dict, Any, Tuple


def get_top_selling_product(sales: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Identify product with highest total quantity sold."""
    # What is used: collections.Counter for quantity tallying.
    # Why it is used: Provides concise counting interface and most_common() extraction.
    # How it works: Iterates sales entries, updating Counter with product quantity counts.
    product_counts: Counter = Counter()

    for sale in sales:
        product = str(sale.get("product", "Unknown")).strip()
        quantity = int(sale.get("quantity", 0))
        product_counts[product] += quantity

    if not product_counts:
        return ("None", 0)

    top_product, top_qty = product_counts.most_common(1)[0]
    return (top_product, top_qty)


if __name__ == "__main__":
    sales_data = [
        {"product": "Laptop", "quantity": 5},
        {"product": "Mouse", "quantity": 20},
        {"product": "Laptop", "quantity": 3},
    ]
    top_product, top_qty = get_top_selling_product(sales_data)
    print(f"Top Product: {top_product} ({top_qty} units)")

    assert top_product == "Mouse", "Top product name mismatched"
    assert top_qty == 20, "Top product quantity mismatched"
    print("[OK] Challenge 4 Passed Successfully!")

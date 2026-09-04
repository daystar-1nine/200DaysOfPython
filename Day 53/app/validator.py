"""
===============================================================================
DAY 53 — DATA VALIDATOR MODULE
===============================================================================
This module enforces business domain validation constraints on cleaned sales records.
===============================================================================
"""

from datetime import date
from typing import Dict, Any


def validate_sale_dict(record: Dict[str, Any]) -> bool:
    """Verify that a cleaned record dictionary satisfies domain integrity constraints."""
    # What is used: Boolean logical validation boundary checks.
    # Why it is used: Filters out mathematically impossible or corrupt sales transactions.
    # How it works: Returns True if order_id > 0, price >= 0, quantity > 0, and fields exist.
    if not isinstance(record, dict):
        return False

    order_id = record.get("order_id")
    if not isinstance(order_id, int) or order_id <= 0:
        return False

    customer = record.get("customer")
    if not isinstance(customer, str) or not customer.strip():
        return False

    product = record.get("product")
    if not isinstance(product, str) or not product.strip():
        return False

    category = record.get("category")
    if not isinstance(category, str) or not category.strip():
        return False

    price = record.get("price")
    if not isinstance(price, (int, float)) or price < 0.0:
        return False

    quantity = record.get("quantity")
    if not isinstance(quantity, int) or quantity <= 0:
        return False

    transaction_date = record.get("date")
    if not isinstance(transaction_date, date):
        return False

    return True

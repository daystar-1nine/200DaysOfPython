"""
===============================================================================
DAY 53 — CODING CHALLENGE 2: REMOVE DUPLICATES
===============================================================================
Topic: Order Record Deduplication
Goal: Identify and filter duplicate order dictionaries from a list based on unique ID.
===============================================================================
"""

from typing import List, Dict, Any


def remove_duplicate_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate order dictionaries preserving original sequence by order ID."""
    # What is used: set() for tracking unique order IDs and list builder.
    # Why it is used: Ensures $O(1)$ duplicate ID lookup performance during filtering.
    # How it works: Appends orders to unique list only if order ID hasn't been seen.
    seen_ids = set()
    unique_orders = []

    for order in orders:
        order_id = order.get("id")
        if order_id not in seen_ids:
            seen_ids.add(order_id)
            unique_orders.append(order)

    return unique_orders


if __name__ == "__main__":
    raw_orders = [
        {"id": 1, "amount": 500},
        {"id": 2, "amount": 800},
        {"id": 1, "amount": 500},
    ]
    result = remove_duplicate_orders(raw_orders)
    print("Raw Orders Count:   ", len(raw_orders))
    print("Unique Orders Count:", len(result))
    print("Deduplicated Orders:", result)

    assert len(result) == 2, f"Expected 2 unique orders, got {len(result)}"
    assert [o["id"] for o in result] == [1, 2], "Order IDs match failed"
    print("[OK] Challenge 2 Passed Successfully!")

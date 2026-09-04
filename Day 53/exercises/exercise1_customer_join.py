"""
===============================================================================
DAY 53 — EXERCISE 1: CUSTOMER DATA JOIN (PYTHON IN-MEMORY INNER JOIN)
===============================================================================
Topic: Relational Data Joining in Pure Python
Goal: Combine sales dataset with customer location dataset on customer name/ID
      to perform regional sales analysis.
===============================================================================
"""

from typing import List, Dict, Any
from collections import defaultdict


def join_sales_and_customers(
    sales: List[Dict[str, Any]], customers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Perform an inner join between sales records and customer metadata."""
    # What is used: Dictionary lookup hash map indexed by customer name.
    # Why it is used: Provides $O(1)$ key lookup time for joining tables.
    # How it works: Maps customer name to city metadata, merging city into sale dict.
    customer_map = {
        c["name"].strip().lower(): c["city"].strip()
        for c in customers
        if "name" in c and "city" in c
    }

    joined_records = []
    for sale in sales:
        customer_name = str(sale.get("customer", "")).strip().lower()
        if customer_name in customer_map:
            joined_item = dict(sale)
            joined_item["city"] = customer_map[customer_name]
            joined_records.append(joined_item)

    return joined_records


def calculate_city_spending(joined_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregate total spending grouped by city."""
    # What is used: collections.defaultdict(float).
    # Why it is used: Accumulates city revenue totals.
    # How it works: Iterates joined records, summing total into city key.
    city_totals: Dict[str, float] = defaultdict(float)
    for row in joined_data:
        city = row.get("city", "Unknown")
        total = float(row.get("total", 0.0))
        city_totals[city] += total
    return dict(city_totals)


if __name__ == "__main__":
    customers_dataset = [
        {"customer_id": 1, "name": "Rahul", "city": "Mumbai"},
        {"customer_id": 2, "name": "Aisha", "city": "Pune"},
        {"customer_id": 3, "name": "Rohan", "city": "Nashik"},
    ]

    sales_dataset = [
        {"order_id": 1001, "customer": "Rahul", "total": 55000.0},
        {"order_id": 1002, "customer": "Aisha", "total": 1200.0},
        {"order_id": 1003, "customer": "Rahul", "total": 70000.0},
        {"order_id": 1004, "customer": "Rohan", "total": 2500.0},
    ]

    joined = join_sales_and_customers(sales_dataset, customers_dataset)
    city_revenue = calculate_city_spending(joined)

    print("Joined Sales Records with City Metadata:")
    for j in joined:
        print(f"  Order #{j['order_id']} | Customer: {j['customer']} | City: {j['city']} | Total: ${j['total']:,.2f}")

    print("\nRegional Revenue Breakdown:")
    for city, rev in city_revenue.items():
        print(f"  {city}: ${rev:,.2f}")

    assert city_revenue["Mumbai"] == 125000.0, "Mumbai total mismatched"
    assert city_revenue["Pune"] == 1200.0, "Pune total mismatched"
    print("\n[OK] Exercise 1 Passed Successfully!")

"""
===============================================================================
DAY 53 — EXERCISE 2: MINI DATA WAREHOUSE MULTI-JOIN
===============================================================================
Topic: 3-Way Relational JOIN in Python (Products + Sales + Customers)
Goal: Replicate a Data Warehouse dimensional modeling join using pure Python.
===============================================================================
"""

from typing import List, Dict, Any
from collections import defaultdict


def perform_warehouse_join(
    sales: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    products: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join transactions with customer dimension and product dimension hash maps."""
    # What is used: Dictionary lookup hash maps for 3-way join execution.
    # Why it is used: Simulates SQL star-schema join operations in pure Python.
    # How it works: Lookups product unit price/category and customer city/state by key.
    customer_dim = {c["id"]: c for c in customers}
    product_dim = {p["id"]: p for p in products}

    fact_table = []
    for sale in sales:
        c_id = sale.get("customer_id")
        p_id = sale.get("product_id")

        if c_id in customer_dim and p_id in product_dim:
            cust = customer_dim[c_id]
            prod = product_dim[p_id]
            qty = sale.get("quantity", 1)
            total = prod["price"] * qty

            fact_table.append({
                "order_id": sale["order_id"],
                "customer": cust["name"],
                "city": cust["city"],
                "product": prod["product"],
                "category": prod["category"],
                "unit_price": prod["price"],
                "quantity": qty,
                "total": total,
            })

    return fact_table


if __name__ == "__main__":
    customers = [
        {"id": 1, "name": "Rahul", "city": "Mumbai"},
        {"id": 2, "name": "Aisha", "city": "Pune"},
    ]

    products = [
        {"id": 101, "product": "Laptop", "category": "Electronics", "price": 55000.0},
        {"id": 102, "product": "Mouse", "category": "Electronics", "price": 1200.0},
        {"id": 103, "product": "Chair", "category": "Furniture", "price": 7000.0},
    ]

    sales = [
        {"order_id": 1001, "customer_id": 1, "product_id": 101, "quantity": 1},
        {"order_id": 1002, "customer_id": 2, "product_id": 102, "quantity": 2},
        {"order_id": 1003, "customer_id": 1, "product_id": 103, "quantity": 2},
    ]

    fact_sales = perform_warehouse_join(sales, customers, products)

    print("Data Warehouse Fact Table Output:")
    for row in fact_sales:
        print(f"  Order #{row['order_id']} | {row['customer']} ({row['city']}) bought {row['quantity']}x {row['product']} ({row['category']}) -> Total: ${row['total']:,.2f}")

    total_revenue = sum(r["total"] for r in fact_sales)
    print(f"\nTotal Fact Table Revenue: ${total_revenue:,.2f}")

    assert len(fact_sales) == 3, "Fact table row count mismatched"
    assert total_revenue == 71400.0, "Total warehouse revenue mismatched"
    print("[OK] Exercise 2 Passed Successfully!")

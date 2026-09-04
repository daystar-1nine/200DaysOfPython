"""
Day 57 - Coding Challenge 2: Top 3 Customers by Total Order Value
Find top 3 customers ordered by total order value.
"""

# What is used: Import pandas library.
# Why it is used: Core package for groupby sum and nlargest operations.
# How it works: Brings pandas namespace into module scope.
import pandas as pd


def get_top_3_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total order value per Customer_ID and return top 3 customers.

    Args:
        df: Input DataFrame containing Customer_ID and Order_Value columns.

    Returns:
        pd.DataFrame: Top 3 customers by total order value.
    """
    # What is used: df.groupby("Customer_ID")["Order_Value"].sum().reset_index().
    # Why it is used: Aggregates order values per customer.
    # How it works: Sums Order_Value per Customer_ID.
    cust_totals = df.groupby("Customer_ID")["Order_Value"].sum().reset_index()

    # What is used: cust_totals.nlargest(3, "Order_Value").
    # Why it is used: Efficiently extracts top 3 rows without full sorting.
    # How it works: Sorts top 3 entries by Order_Value in descending order.
    top_3 = cust_totals.nlargest(3, "Order_Value")
    return top_3


if __name__ == "__main__":
    # What is used: Dictionary defining mock customer order history.
    # Why it is used: Serves as test data for customer aggregation.
    # How it works: Maps Customer_ID and Order_Value entries.
    orders = pd.DataFrame({
        "Customer_ID": ["C01", "C02", "C01", "C03", "C02", "C04", "C03"],
        "Order_Value": [500, 800, 300, 900, 400, 150, 1200]
    })

    # What is used: Calling get_top_3_customers.
    # Why it is used: Obtains top 3 high-value customers.
    # How it works: Prints top 3 customer records.
    top3_df = get_top_3_customers(orders)
    print("--- Top 3 Customers by Total Order Value ---")
    print(top3_df)

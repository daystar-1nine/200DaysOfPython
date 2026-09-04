"""
Day 56 - Coding Challenge 5: Product Sales Revenue Calculator
Calculate line-item revenue (Quantity * Price) and aggregate total revenue metrics.
"""

# What is used: Import pandas library.
# Why it is used: Essential for financial table operations and aggregate summaries.
# How it works: Imports pandas module into namespace.
import pandas as pd


def analyze_product_revenue(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Compute product line revenue and total sales revenue.

    Args:
        df: Input DataFrame with Quantity and Unit_Price columns.

    Returns:
        tuple[pd.DataFrame, float]: Augmented DataFrame with Revenue column and total revenue sum.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures input DataFrame remains unmodified.
    # How it works: Duplicates memory representation of the DataFrame.
    sales_df = df.copy()

    # What is used: Vectorized multiplication df["Quantity"] * df["Unit_Price"].
    # Why it is used: Calculates total line item revenue in a single SIMD operation.
    # How it works: Multiplies elements at identical row positions across Series.
    sales_df["Revenue"] = sales_df["Quantity"] * sales_df["Unit_Price"]

    # What is used: sales_df["Revenue"].sum() aggregation.
    # Why it is used: Obtains grand total sales revenue for all catalog products.
    # How it works: Sums all values in Revenue column into a float scalar.
    total_revenue = float(sales_df["Revenue"].sum())

    return sales_df, total_revenue


if __name__ == "__main__":
    # What is used: Dictionary defining mock e-commerce sales dataset.
    # Why it is used: Provides catalog sales data for testing.
    # How it works: Maps product details to lists.
    catalog = {
        "Product_ID": ["P101", "P102", "P103", "P104"],
        "Product_Name": ["Laptop", "Mouse", "Keyboard", "Monitor"],
        "Quantity": [5, 50, 20, 10],
        "Unit_Price": [1200.0, 25.0, 45.0, 300.0]
    }

    # What is used: pd.DataFrame constructor.
    # Why it is used: Creates Pandas DataFrame from catalog dictionary.
    # How it works: Initializes tabular memory structure.
    df_sales = pd.DataFrame(catalog)

    # What is used: analyze_product_revenue call.
    # Why it is used: Computes product line revenues and total revenue sum.
    # How it works: Unpacks DataFrame and total scalar float value.
    result_df, total = analyze_product_revenue(df_sales)
    print("--- Product Revenue Analysis ---")
    print(result_df)
    print(f"\nTotal Grand Revenue: ${total:,.2f}")

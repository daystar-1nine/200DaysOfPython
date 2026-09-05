"""
Data Preprocessing & Segment Derivation Module
=============================================
Performs date coercion, type casting, and deterministic customer segmentation.
"""

import pandas as pd
import numpy as np

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transactions and derives analytical dimensions.

    # What is used: pd.to_datetime, numeric type casting, and deterministic segmentation
    # Why it is used: Ensures uniform data structures required for statistical plotting
    # How it works: Parses timestamps, computes discount percentages, and maps customer segments
    """
    data = df.copy()

    # Coerce timestamps
    data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")

    # Ensure numeric columns are strictly float/int
    num_cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Cost", "Profit", "Profit_Margin"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Derive Customer_Segment deterministically if absent
    if "Customer_Segment" not in data.columns:
        # Deterministic assignment based on Customer_ID hash for consistency
        def assign_segment(cust_id):
            h = hash(str(cust_id)) % 3
            if h == 0:
                return "Consumer"
            elif h == 1:
                return "Corporate"
            else:
                return "Home Office"

        data["Customer_Segment"] = data["Customer_ID"].apply(assign_segment)

    # Derive Discount_Percent for intuitive visualization
    data["Discount_Percent"] = data["Discount"] * 100.0

    return data

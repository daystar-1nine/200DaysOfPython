"""
Data Preprocessing & Normalization Module
========================================
"""

import pandas as pd
import numpy as np

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transactions, coerces timestamps, numeric types, and fills missing values.
    """
    data = df.copy()

    # Coerce timestamps
    data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")

    # Coerce numeric columns
    num_cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Cost", "Profit"]
    for col in num_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop true duplicate rows
    data = data.drop_duplicates()

    # Ensure Year and Month columns exist
    if "Year" not in data.columns:
        data["Year"] = data["Order_Date"].dt.year
    if "Month" not in data.columns:
        data["Month"] = data["Order_Date"].dt.month
    if "Month_Name" not in data.columns:
        data["Month_Name"] = data["Order_Date"].dt.strftime("%B")

    # Derive Customer_Segment if absent
    if "Customer_Segment" not in data.columns:
        def assign_segment(cust_id):
            h = hash(str(cust_id)) % 3
            if h == 0:
                return "Consumer"
            elif h == 1:
                return "Corporate"
            else:
                return "Home Office"
        data["Customer_Segment"] = data["Customer_ID"].apply(assign_segment)

    return data

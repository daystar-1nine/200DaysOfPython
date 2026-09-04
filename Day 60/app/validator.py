"""
Module: validator.py
Audits cleaned sales data against core business rules and domain boundaries for Day 60 BI Analytics Engine.
"""

# What is used: Import pandas library.
# Why it is used: Executes vectorized validation rule conditions.
# How it works: Checks uniqueness, non-negativity, discount boundaries, and region categories.
import pandas as pd
from app.config import ALLOWED_REGIONS


def validate_sales_data(df: pd.DataFrame) -> dict:
    """
    Validate sales dataset against business domain rules.

    Validation Rules:
    1. Order_ID uniqueness (0 duplicates).
    2. Quantity positivity (> 0).
    3. Unit_Price and Cost_Price non-negativity (>= 0).
    4. Discount validity (0.0 <= Discount <= 1.0).
    5. Valid Region categorization.
    6. Non-null Order_Date.

    Args:
        df: Cleaned sales DataFrame.

    Returns:
        dict: Validation audit report containing rule-by-rule status and overall is_valid flag.
    """
    rules = {}

    # Rule 1: Order_ID Uniqueness
    dup_orders = int(df["Order_ID"].duplicated().sum()) if "Order_ID" in df.columns else 0
    rules["unique_order_ids"] = {"passed": dup_orders == 0, "violations": dup_orders}

    # Rule 2: Quantity > 0
    invalid_qty = int((df["Quantity"] <= 0).sum()) if "Quantity" in df.columns else 0
    rules["positive_quantity"] = {"passed": invalid_qty == 0, "violations": invalid_qty}

    # Rule 3: Prices >= 0
    invalid_price = int(((df["Unit_Price"] < 0) | (df["Cost_Price"] < 0)).sum()) if ("Unit_Price" in df.columns and "Cost_Price" in df.columns) else 0
    rules["non_negative_prices"] = {"passed": invalid_price == 0, "violations": invalid_price}

    # Rule 4: 0.0 <= Discount <= 1.0
    invalid_disc = int((~df["Discount"].between(0.0, 1.0)).sum()) if "Discount" in df.columns else 0
    rules["valid_discount_range"] = {"passed": invalid_disc == 0, "violations": invalid_disc}

    # Rule 5: Allowed Regions
    invalid_regions = int((~df["Region"].isin(ALLOWED_REGIONS)).sum()) if "Region" in df.columns else 0
    rules["valid_regions"] = {"passed": invalid_regions == 0, "violations": invalid_regions}

    # Rule 6: Non-null Date
    null_dates = int(df["Order_Date"].isna().sum()) if "Order_Date" in df.columns else 0
    rules["valid_dates"] = {"passed": null_dates == 0, "violations": null_dates}

    all_passed = all(r["passed"] for r in rules.values())

    return {
        "is_valid": all_passed,
        "rules": rules
    }

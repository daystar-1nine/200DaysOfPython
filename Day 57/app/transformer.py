"""
Module: transformer.py
Performs revenue calculation, datetime parsing, Month period extraction, and discount range binning.
"""

# What is used: Import pandas library.
# Why it is used: Core package for data transformation, datetime handling, and pd.cut binning.
# How it works: Brings pandas namespace into execution context.
import pandas as pd


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform sales DataFrame by calculating Revenue, parsing Order_Date into datetime,
    extracting Month period, and binning Discount into ranges.

    Formula: Revenue = Quantity * Unit_Price * (1 - Discount)

    Args:
        df: Cleaned sales DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame augmented with calculated columns.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures input DataFrame is not mutated directly.
    # How it works: Duplicates DataFrame memory buffer.
    transformed_df = df.copy()

    # What is used: Vectorized arithmetic multiplication Quantity * Unit_Price * (1 - Discount).
    # Why it is used: Computes net sales revenue per order line item.
    # How it works: Multiplies elementwise across Series vectors.
    transformed_df["Revenue"] = (
        transformed_df["Quantity"]
        * transformed_df["Unit_Price"]
        * (1.0 - transformed_df["Discount"])
    ).round(2)

    # What is used: pd.to_datetime() conversion.
    # Why it is used: Converts date strings into native Pandas Timestamp datetime objects.
    # How it works: Parses string formats into datetime64[ns] dtype.
    transformed_df["Order_Date"] = pd.to_datetime(transformed_df["Order_Date"], errors="coerce")

    # What is used: Datetime accessor .dt.to_period("M").
    # Why it is used: Extracts monthly period (e.g., '2026-01') for time-series aggregation.
    # How it works: Converts datetime timestamps into monthly period index representations.
    transformed_df["Month"] = transformed_df["Order_Date"].dt.to_period("M").astype(str)

    # What is used: pd.cut() for discount range binning.
    # Why it is used: Categorizes continuous discount rates into discrete analytical tiers.
    # How it works: Maps Discount values to bins [0.0, 0.05, 0.15, 1.0] with labels.
    bins = [-0.01, 0.0, 0.09, 0.19, 1.0]
    labels = ["No Discount (0%)", "Low (1-9%)", "Medium (10-19%)", "High (20%+)"]
    transformed_df["Discount_Range"] = pd.cut(
        transformed_df["Discount"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return transformed_df

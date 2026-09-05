"""
Dataset Loader & Validation Module
=================================
Safely ingests sales transaction data from CSV files and verifies schema integrity.
"""

import os
import pandas as pd
from app.config import DATA_PATH

REQUIRED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_ID", "Customer_Name",
    "Region", "Category", "Product", "Quantity", "Unit_Price",
    "Cost_Price", "Discount", "Revenue", "Cost", "Profit", "Profit_Margin"
]

def load_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Loads and validates the e-commerce sales CSV dataset.

    # What is used: os.path verification and pd.read_csv with schema validation
    # Why it is used: Guards against missing files and malformed schema definitions
    # How it works: Checks file existence, loads into DataFrame, and checks required columns
    """
    target_path = filepath if filepath is not None else DATA_PATH

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")

    df = pd.read_csv(target_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df

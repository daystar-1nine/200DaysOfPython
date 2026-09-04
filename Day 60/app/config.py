"""
Module: config.py
Configuration constants, schema specifications, and path definitions for Day 60 BI Analytics Engine.
"""

# What is used: Import pathlib Path class.
# Why it is used: Resolves project root and standard directory locations.
# How it works: Derives directory locations relative to config.py.
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sales.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_sales.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

REQUIRED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_ID", "Customer_Name",
    "Region", "Category", "Product", "Quantity", "Unit_Price",
    "Cost_Price", "Discount"
]

NUMERIC_COLUMNS = ["Quantity", "Unit_Price", "Cost_Price", "Discount"]
ALLOWED_REGIONS = {"North", "South", "East", "West"}
